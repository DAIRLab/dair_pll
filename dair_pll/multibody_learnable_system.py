"""Construction and analysis of learnable multibody systems.

Similar to Drake, multibody systems are instantiated as a child class of
:py:class:`System`: :py:class:`MultibodyLearnableSystem`. This object is a thin
wrapper for a :py:class:`MultibodyTerms` member variable, which manages
computation of lumped terms necessary for simulation and evaluation.

Simulation is implemented via Anitescu's [1] convex method.

An interface for the ContactNets [2] loss is also defined as an alternative
to prediction loss.

A large portion of the internal implementation of :py:class:`DrakeSystem` is
implemented in :py:class:`MultibodyPlantDiagram`.

[1] M. Anitescu, “Optimization-based simulation of nonsmooth rigid
multibody dynamics,” Mathematical Programming, 2006,
https://doi.org/10.1007/s10107-005-0590-7

[2] S. Pfrommer*, M. Halm*, and M. Posa. "ContactNets: Learning Discontinuous
Contact Dynamics with Smooth, Implicit Representations," Conference on
Robotic Learning, 2020, https://proceedings.mlr.press/v155/pfrommer21a.html
"""
from multiprocessing import pool
from os import path
from typing import List, Tuple, Optional, Dict, cast

import numpy as np
import torch
import pdb
import time
# from sappy import SAPSolver  # type: ignore
from torch import Tensor
from torch.nn import Module
import torch.nn as nn

from dair_pll import urdf_utils, tensor_utils, file_utils
from dair_pll.drake_system import DrakeSystem
from dair_pll.integrator import VelocityIntegrator
from dair_pll.multibody_terms import MultibodyTerms
from dair_pll.quaternion import quaternion_to_rotmat_vec
from dair_pll.solvers import DynamicCvxpyLCQPLayer
from dair_pll.state_space import FloatingBaseSpace
from dair_pll.system import System, SystemSummary
from dair_pll.tensor_utils import pbmm, broadcast_lorentz, \
    one_vector_block_diagonal, project_lorentz, reflect_lorentz


# Loss variations options
LOSS_PLL_ORIGINAL = 'loss_pll_original'
LOSS_POWER = 'loss_power'
LOSS_INERTIA_AGNOSTIC = 'loss_inertia_agnostic'
LOSS_BALANCED = 'loss_balanced'
LOSS_CONTACT_VELOCITY = 'loss_contact_velocity'
LOSS_VARIATIONS = [LOSS_PLL_ORIGINAL, LOSS_POWER, LOSS_INERTIA_AGNOSTIC,
                   LOSS_BALANCED, LOSS_CONTACT_VELOCITY]
LOSS_VARIATION_NUMBERS = [str(LOSS_VARIATIONS.index(loss_variation)) \
                          for loss_variation in LOSS_VARIATIONS]

# Scaling factors to equalize translation and rotation errors.
# For rotation versus linear scaling:  penalize 0.1 meters same as 90 degrees.
ROTATION_SCALING = 0.2/torch.pi
# For articulation versus linear/rotation scaling:  penalize the scenario where
# one elbow link is in the right place and the other is 180 degrees flipped the
# same, whether link 1 or link 2 are in the right place.
ELBOW_COM_TO_AXIS_DISTANCE = 0.035
JOINT_SCALING = 2*ELBOW_COM_TO_AXIS_DISTANCE/torch.pi + ROTATION_SCALING


class MultibodyLearnableSystem(System):
    """:py:class:`System` interface for dynamics associated with
    :py:class:`MultibodyTerms`."""
    multibody_terms: MultibodyTerms
    init_urdfs: Dict[str, str]
    output_urdfs_dir: Optional[str] = None
    visualization_system: Optional[DrakeSystem]
    solver: DynamicCvxpyLCQPLayer
    dt: float
    loss_variation_txt: str

    def __init__(self,
                 init_urdfs: Dict[str, str],
                 dt: float,
                 inertia_mode: int,
                 loss_variation: int,
                 w_pred: float,
                 w_comp: float,
                 w_diss: float,
                 w_pen: float,
                 w_res: float,
                 w_res_w: float,
                 do_residual: bool = False,
                 output_urdfs_dir: Optional[str] = None,
                 network_width: int = 128,
                 network_depth: int = 2,
                 represent_geometry_as: str = 'box',
                 randomize_initialization: bool = False,
                 g_frac: float = 1.0) -> None:
        """Inits :py:class:`MultibodyLearnableSystem` with provided model URDFs.

        Implementation is primarily based on Drake. Bodies are modeled via
        :py:class:`MultibodyTerms`, which uses Drake symbolics to generate
        dynamics terms, and the system can be exported back to a
        Drake-interpretable representation as a set of URDFs.

        Args:
            init_urdfs: Names and corresponding URDFs to model with
              :py:class:`MultibodyTerms`.
            dt: Time step of system in seconds.
            inertia_mode: An integer 0, 1, 2, 3, or 4 representing the
              inertial parameters the model can learn.  The higher the number
              the more inertial parameters are free to be learned, and 0
              corresponds to learning no inertial parameters.
            loss_variation: An integer 0, 1, 2, 3, or 4 representing the loss
              variation to use. 0 indicates the original PLL loss, 1 power loss,
              2 inertia-agnostic, 3 balanced inertia-agnostic, and 4 contact
              velocity inertia-agnostic.
            output_urdfs_dir: Optionally, a directory that learned URDFs can be
              written to.
            randomize_initialization: Whether to randomize and export the
              initialization or not.
        """
        assert str(loss_variation) in LOSS_VARIATION_NUMBERS

        multibody_terms = MultibodyTerms(init_urdfs, inertia_mode,
                                         represent_geometry_as,
                                         randomize_initialization,
                                         g_frac=g_frac)

        space = multibody_terms.plant_diagram.space
        integrator = VelocityIntegrator(space, self.sim_step, dt)
        super().__init__(space, integrator)
        
        self.output_urdfs_dir = output_urdfs_dir
        self.multibody_terms = multibody_terms
        self.init_urdfs = init_urdfs

        if randomize_initialization:
            # Add noise and export.
            print(f'Randomizing initialization.')
            multibody_terms.randomize_multibody_terms(inertia_mode)
            self.multibody_terms = multibody_terms
            self.generate_updated_urdfs('init')

        self.loss_variation_txt = LOSS_VARIATIONS[loss_variation]
        self.visualization_system = None
        self.solver = DynamicCvxpyLCQPLayer(self.space.n_v)
        self.dt = dt
        self.set_carry_sampler(lambda: Tensor([False]))
        self.max_batch_dim = 1
        self.w_pred = w_pred
        self.w_comp = w_comp
        self.w_diss = w_diss
        self.w_pen = w_pen
        self.w_res = w_res
        self.w_res_w = w_res_w

        self.residual_net = None

        if do_residual:
            # This system type is only well defined for systems containing a
            # fixed ground and one floating base system.
            assert len(self.space.spaces) == 2
            self.object_space_idx = None
            for idx in range(len(self.space.spaces)):
                if type(self.space.spaces[idx]) == FloatingBaseSpace:
                    self.object_space_idx = idx
            assert self.object_space_idx != None

            self.init_residual_network(network_width, network_depth)

    def generate_updated_urdfs(self, suffix: str = None) -> Dict[str, str]:
        """Exports current parameterization as a :py:class:`DrakeSystem`.

        Returns:
            New Drake system instantiated on new URDFs.
        """
        assert self.output_urdfs_dir is not None
        old_urdfs = self.init_urdfs
        new_urdf_strings = urdf_utils.represent_multibody_terms_as_urdfs(
            self.multibody_terms, self.output_urdfs_dir)
        new_urdfs = {}

        # saves new urdfs with original file basenames plus optional suffix in
        # new folder.
        for urdf_name, new_urdf_string in new_urdf_strings.items():
            new_urdf_filename = path.basename(old_urdfs[urdf_name])
            if suffix != None:
                new_urdf_filename = new_urdf_filename.split('.')[0] + '_' + \
                                    suffix + '.urdf'

            new_urdf_path = path.join(self.output_urdfs_dir, new_urdf_filename)
            file_utils.save_string(new_urdf_path, new_urdf_string)
            new_urdfs[urdf_name] = new_urdf_path

        return new_urdfs

    def contactnets_loss(self,
                         x: Tensor,
                         u: Tensor,
                         x_plus: Tensor,
                         loss_pool: Optional[pool.Pool] = None) -> Tensor:
        r"""Calculate ContactNets [1] loss for state transition.

        Change made to scale this loss to be per kilogram.  This helps prevent
        sending mass quantities to zero in multibody learning scenarios.

        References:
            [1] S. Pfrommer*, M. Halm*, and M. Posa. "ContactNets: Learning
            Discontinuous Contact Dynamics with Smooth, Implicit
            Representations," Conference on Robotic Learning, 2020,
            https://proceedings.mlr.press/v155/pfrommer21a.html

        Args:
            x: (\*, space.n_x) current state batch.
            u: (\*, ?) input batch.
            x_plus: (\*, space.n_x) current state batch.
            loss_pool: optional processing pool to enable multithreaded solves.

        Returns:
            (\*,) loss batch.
        """
        loss_pred, loss_comp, loss_pen, loss_diss = \
            self.calculate_contactnets_loss_terms(x, u, x_plus)

        regularizers = self.get_regularization_terms(x, u, x_plus)

        # For now just take the first regularization term as the residual
        # regularization.  Will need to be updated later if more are added.
        # residual_norm = regularizers[0]
        # regs = self.w_res * residual_norm
        reg_norm = regularizers[0]
        reg_weight = regularizers[1]

        loss = (self.w_res * reg_norm) + (self.w_res_w * reg_weight) + \
               (self.w_pred * loss_pred) + (self.w_comp * loss_comp) + \
               (self.w_pen * loss_pen) + (self.w_diss * loss_diss)

        return loss

    def get_regularization_terms(self, x: Tensor, u: Tensor,
                                 x_plus: Tensor) -> List[Tensor]:
        """Calculate some regularization terms."""

        regularizers = []

        # Residual size regularization.
        if self.residual_net != None:
            # Penalize the size of the residual.  Good with w_res = 0.01.
            residual = self.residual_net(x_plus)
            residual_norm = torch.linalg.norm(residual, dim=1) ** 2
            regularizers.append(residual_norm)

            # Additionally penalize the residual network weights.  This will get
            # scaled down to approximately the same size as the residual norm.
            l2_penalty = torch.zeros((x.shape[0],))
            for layer in self.residual_net:
                if isinstance(layer, nn.Linear):
                    l2_penalty += sum([(p**2).sum() for p in layer.weight])
            # l2_penalty *= 1e-3

            regularizers.append(l2_penalty)

        else:
            # Otherwise, append 0 twice for the residual norm and weights.
            regularizers.append(torch.zeros((x.shape[-2],)))
            regularizers.append(torch.zeros((x.shape[-2],)))

        # TODO: Use the believed geometry to help supervise the learned CoM.
        # if (self.multibody_terms.inertia_mode_txt != 'none') and \
        #    (self.multibody_terms.inertia_mode_txt != 'masses'):
        #     # This means the CoM locations are getting learned.
        #     pass

        return regularizers

    def calculate_contactnets_loss_terms(self,
                         x: Tensor,
                         u: Tensor,
                         x_plus: Tensor) -> \
                         Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Helper function for
        :py:meth:`MultibodyLearnableSystem.contactnets_loss` that returns the
        individual pre-weighted loss contributions:

            * Prediction
            * Complementarity
            * Penetration
            * Dissipation

        Args:
            x: (*, space.n_x) current state batch.
            u: (*, ?) input batch.
            x_plus: (*, space.n_x) current state batch.

        Returns:
            (*,) prediction error loss.
            (*,) complementarity violation loss.
            (*,) penetration loss.
            (*,) dissipation violation loss.
            (*,) residual regularizer (0 if no residual).
        """
        # pylint: disable-msg=too-many-locals
        v = self.space.v(x)
        q_plus, v_plus = self.space.q_v(x_plus)
        dt = self.dt
        # eps = 1e-3

        # Begin loss calculation.
        delassus, M, J, phi, non_contact_acceleration = \
            self.get_multibody_terms(q_plus, v_plus, u)

        try:
            M_inv = torch.inverse((M))
        except:
            print(f'M: {M}')
            pdb.set_trace()

        # Construct a reordering matrix s.t. lambda_CN = reorder_mat @ f_sappy.
        n_contacts = phi.shape[-1]
        reorder_mat = tensor_utils.sappy_reorder_mat(n_contacts)
        reorder_mat = reorder_mat.reshape((1,) * (delassus.dim() - 2) +
                                          reorder_mat.shape).expand(
                                              delassus.shape)
        J_t = J[..., n_contacts:, :]

        # Construct a diagonal scaling matrix (3*n_contacts, 3*n_contacts) S
        # s.t. S @ lambda_CN = scaled lambdas in units [m/s] instead of [N s].
        delassus_diag_vec = torch.diagonal(delassus, dim1=-2, dim2=-1)
        contact_weights = pbmm(one_vector_block_diagonal(n_contacts, 3).t(),
                               pbmm(reorder_mat.transpose(-1, -2),
                                    delassus_diag_vec.unsqueeze(-1)))
        contact_weights = broadcast_lorentz(contact_weights.squeeze(-1))
        S = torch.diag_embed(contact_weights)

        # Construct a diagonal scaling matrix (n_velocity, n_velocity) P s.t.
        # velocity errors are scaled to relate translation and rotation errors
        # in a thoughful way.
        P_diag = torch.ones_like(v)
        P_diag[..., :3] *= ROTATION_SCALING
        P_diag[..., 6:] *= JOINT_SCALING
        P = torch.diag_embed(P_diag)

        # pylint: disable=E1103
        double_zero_vector = torch.zeros(phi.shape[:-1] + (2 * n_contacts,))
        phi_then_zero = torch.cat((phi, double_zero_vector), dim=-1)

        # pylint: disable=E1103
        sliding_velocities = pbmm(J_t, v_plus.unsqueeze(-1))
        sliding_speeds = sliding_velocities.reshape(phi.shape[:-1] +
                                                    (n_contacts, 2)).norm(
                                                        dim=-1, keepdim=True)

        # Calculate "half delassus" based on loss formulation mode.
        if self.loss_variation_txt == LOSS_PLL_ORIGINAL:
            L = torch.linalg.cholesky(M_inv)
            half_delassus = pbmm(J, L)
        elif self.loss_variation_txt == LOSS_POWER:
            L = torch.linalg.cholesky(M_inv)
            half_delassus = pbmm(J, L)
        elif self.loss_variation_txt == LOSS_INERTIA_AGNOSTIC:
            half_delassus = pbmm(J, M_inv)
        elif self.loss_variation_txt == LOSS_BALANCED:
            half_delassus = pbmm(pbmm(J, M_inv), P)
        elif self.loss_variation_txt == LOSS_CONTACT_VELOCITY:
            half_delassus = delassus

        Q = pbmm(half_delassus, half_delassus.transpose(-1, -2)) #+ \
            #eps * torch.eye(3 * n_contacts)

        J_M = pbmm(reorder_mat.transpose(-1,-2), half_delassus)

        dv = (v_plus - (v + non_contact_acceleration * dt)).unsqueeze(-2)

        # Calculate q vectors based on loss formulation mode.
        if self.loss_variation_txt == LOSS_PLL_ORIGINAL:
            q_pred = -pbmm(J, dv.transpose(-1, -2))
            q_comp = torch.abs(phi_then_zero).unsqueeze(-1)
            q_diss = dt*torch.cat((sliding_speeds, sliding_velocities), dim=-2)
        elif self.loss_variation_txt == LOSS_POWER:
            q_pred = -pbmm(J, dv.transpose(-1, -2))
            q_comp = (1/dt) * torch.abs(phi_then_zero).unsqueeze(-1)
            q_diss = torch.cat((sliding_speeds, sliding_velocities), dim=-2)
        elif self.loss_variation_txt == LOSS_INERTIA_AGNOSTIC:
            q_pred = -pbmm(J, pbmm(M_inv, dv.transpose(-1, -2)))
            # q_comp = (1/dt) * torch.abs(phi_then_zero).unsqueeze(-1)
            # q_diss = torch.cat((sliding_speeds, sliding_velocities), dim=-2)
            q_comp = (1/dt) * pbmm(S, torch.abs(phi_then_zero).unsqueeze(-1))
            q_diss = pbmm(S, torch.cat((sliding_speeds, sliding_velocities),
                          dim=-2))
        elif self.loss_variation_txt == LOSS_BALANCED:
            q_pred = -pbmm(J, pbmm(M_inv, pbmm(pbmm(P, P), 
                                               dv.transpose(-1, -2))))
            q_comp = (1/dt) * pbmm(S, torch.abs(phi_then_zero).unsqueeze(-1))
            q_diss = pbmm(S, torch.cat((sliding_speeds, sliding_velocities),
                          dim=-2))
        elif self.loss_variation_txt == LOSS_CONTACT_VELOCITY:
            q_pred = -pbmm(delassus, pbmm(J, dv.transpose(-1, -2)))
            q_comp = (1/dt) * pbmm(S, torch.abs(phi_then_zero).unsqueeze(-1))
            q_diss = pbmm(S, torch.cat((sliding_speeds, sliding_velocities),
                          dim=-2))

        q = q_pred + (self.w_comp/self.w_pred)*q_comp + \
                     (self.w_diss/self.w_pred)*q_diss

        constant_pen = (torch.maximum(
                            -phi, torch.zeros_like(phi))**2).sum(dim=-1)
        constant_pen = constant_pen.reshape(constant_pen.shape + (1,1))

        # Calculate the prediction constant based on loss formulation mode.
        if self.loss_variation_txt == LOSS_PLL_ORIGINAL:
            constant_pred = 0.5 * pbmm(dv, pbmm(M, dv.transpose(-1, -2)))
        elif self.loss_variation_txt == LOSS_POWER:
            constant_pred = 0.5 * pbmm(dv, pbmm(M, dv.transpose(-1, -2)))
        elif self.loss_variation_txt == LOSS_INERTIA_AGNOSTIC:
            constant_pred = 0.5 * pbmm(dv, dv.transpose(-1, -2))
        elif self.loss_variation_txt == LOSS_BALANCED:
            balanced_dv = pbmm(dv, P)
            constant_pred = 0.5 * pbmm(balanced_dv,
                                       balanced_dv.transpose(-1, -2))
        elif self.loss_variation_txt == LOSS_CONTACT_VELOCITY:
            contact_dv = pbmm(dv, J.transpose(-1, -2))
            constant_pred = 0.5 * pbmm(contact_dv, contact_dv.transpose(-1, -2))

        # Envelope theorem guarantees that gradient of loss w.r.t. parameters
        # can ignore the gradient of the force w.r.t. the QCQP parameters.
        # Therefore, we can detach ``force`` from pytorch's computation graph
        # without causing error in the overall loss gradient.
        # pylint: disable=E1103
        try:
            force = pbmm(
                reorder_mat,
                self.solver(  #.apply(
                    J_M,
                    pbmm(reorder_mat.transpose(-1, -2),
                         q).squeeze(-1)).detach().unsqueeze(-1))
                    # pbmm(reorder_mat.transpose(-1, -2), q).squeeze(-1),
                    #eps).detach().unsqueeze(-1))
        except:
            print(f'J_M: {J_M}')
            print(f'reordered q: {pbmm(reorder_mat.transpose(-1, -2), q)}')
            pdb.set_trace()

        # Hack: remove elements of ``force`` where solver likely failed.
        invalid = torch.any((force.abs() > 1e3) | force.isnan() | force.isinf(),
                            dim=-2,
                            keepdim=True)

        constant_pen[invalid] *= 0.
        constant_pred[invalid] *= 0.
        force[invalid.expand(force.shape)] = 0.

        loss_pred = 0.5 * pbmm(force.transpose(-1, -2), pbmm(Q, force)) \
                    + pbmm(force.transpose(-1, -2), q_pred) + constant_pred
        loss_comp = pbmm(force.transpose(-1, -2), q_comp)
        loss_pen = constant_pen
        loss_diss = pbmm(force.transpose(-1, -2), q_diss)

        return loss_pred.reshape(-1), loss_comp.reshape(-1), \
               loss_pen.reshape(-1), loss_diss.reshape(-1)

    def get_multibody_terms(self, q: Tensor, v: Tensor,
        u: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Get multibody terms of the system.  Without a residual, this is a
        straightfoward pass-through to the system's :py:class:`MultibodyTerms`.
        With a residual, the residual augments the continuous dynamics."""

        delassus, M, J, phi, non_contact_acceleration = self.multibody_terms(
            q, v, u)

        if self.residual_net != None:
            # Get the residual network's contribution.
            x = torch.cat((q, v), dim=1)
            residual = self.residual_net(x)/self.dt
            amended_acceleration = non_contact_acceleration + residual

        else:
            amended_acceleration = non_contact_acceleration

        return delassus, M, J, phi, amended_acceleration

    def init_residual_network(self, network_width: int, network_depth: int
        ) -> None:
        """Create and store a neural network architecture that has the multibody
        system state as input and outputs the size of the multibody system's
        velocity space."""

        def make_small_linear_layer(input_size, output_size):
            layer_with_small_init = nn.Linear(input_size, output_size)
            layer_with_small_init.weight.data *= 1e-2
            layer_with_small_init.bias.data *= 1e-2
            return layer_with_small_init

        layers: List[Module] = []

        layers.append(DeepStateAugment3D())

        n_augmented_state = self.space.n_x - 4 + 9
        layers.append(make_small_linear_layer(n_augmented_state, network_width))
        layers.append(nn.ReLU())

        for _ in range(network_depth - 1):
            layers.append(make_small_linear_layer(network_width, network_width))
            layers.append(nn.ReLU())

        layers.append(make_small_linear_layer(network_width, self.space.n_v))

        self.residual_net = nn.Sequential(*layers)

    def forward_dynamics(self,
                         q: Tensor,
                         v: Tensor,
                         u: Tensor,
                         dynamics_pool: Optional[pool.Pool] = None) -> Tensor:
        r"""Calculates delta velocity from current state and input.

        Implements Anitescu's [1] convex formulation in dual form, derived
        similarly to Tedrake [2] and described here.

        Let v_minus be the contact-free next velocity, i.e.::

            v + dt * non_contact_acceleration.

        Let FC be the combined friction cone::

            FC = {[beta_n beta_t]: beta_n_i >= ||beta_t_i||}.

        The primal version of Anitescu's formulation is as follows::

            min_{v_plus,s}  (v_plus - v_minus)^T M(q)(v_plus - v_minus)/2
            s.t.            s = [I; 0]phi(q)/dt + J(q)v_plus,
                            s \\in FC.

        The KKT conditions are the mixed cone complementarity
        problem [3, Theorem 2]::

            s = [I; 0]phi(q)/dt + J(q)v_plus,
            M(q)(v_plus - v_minus) = J(q)^T f,
            FC \\ni s \\perp f \\in FC.

        As M(q) is positive definite, we can solve for v_plus in terms of
        lambda, and thus these conditions can be simplified to::

            FC \\ni D(q)f + J(q)v_minus + [I;0]phi(q)/dt \\perp f \\in FC.

        which in turn are the KKT conditions for the dual QCQP we solve::

            min_{f}     f^T D(q) f/2 + f^T(J(q)v_minus + [I;0]phi(q)/dt)
            s.t.        f \\in FC.

        References:
            [1] M. Anitescu, “Optimization-based simulation of nonsmooth rigid
            multibody dynamics,” Mathematical Programming, 2006,
            https://doi.org/10.1007/s10107-005-0590-7

            [2] R. Tedrake. Underactuated Robotics: Algorithms for Walking,
            Running, Swimming, Flying, and Manipulation (Course Notes for MIT
            6.832), https://underactuated.mit.edu

            [3] S. Z. N'emeth, G. Zhang, "Conic optimization and
            complementarity problems," arXiv,
            https://doi.org/10.48550/arXiv.1607.05161
        Args:
            q: (\*, space.n_q) current configuration batch.
            v: (\*, space.n_v) current velocity batch.
            u: (\*, ?) current input batch.
            dynamics_pool: optional processing pool to enable multithreaded
              solves.

        Returns:
            (\*, space.n_v) delta velocity batch.
        """
        # pylint: disable=too-many-locals
        dt = self.dt
        eps = 1e6
        delassus, M, J, phi, non_contact_acceleration = \
            self.get_multibody_terms(q, v, u)
        n_contacts = phi.shape[-1]
        contact_filter = (broadcast_lorentz(phi) <= eps).unsqueeze(-1)
        contact_matrix_filter = pbmm(contact_filter.int(),
                                     contact_filter.transpose(-1,
                                                              -2).int()).bool()

        reorder_mat = tensor_utils.sappy_reorder_mat(n_contacts)
        reorder_mat = reorder_mat.reshape((1,) * (delassus.dim() - 2) +
                                          reorder_mat.shape).expand(
                                              delassus.shape)

        try:
            L = torch.linalg.cholesky(torch.inverse((M)))
        except:
            try:
                min_eig_val = min(torch.linalg.eigvals(M))
                print(f'\nIssue with M: min eigenvalue of {min_eig_val}')
            except:
                pdb.set_trace()
                print(f'\nCannot calculate eigenvalues of M: {M}')

        J_M = pbmm(reorder_mat.transpose(-1, -2), pbmm(J, L))

        # pylint: disable=E1103
        double_zero_vector = torch.zeros(phi.shape[:-1] + (2 * n_contacts,))
        phi_then_zero = torch.cat((phi, double_zero_vector),
                                  dim=-1).unsqueeze(-1)
        # pylint: disable=E1103
        Q_full = delassus + torch.eye(3 * n_contacts) * 1e-4

        v_minus = v + dt * non_contact_acceleration
        q_full = pbmm(J, v_minus.unsqueeze(-1)) + (1 / dt) * phi_then_zero

        Q = torch.zeros_like(Q_full)
        q = torch.zeros_like(q_full)
        Q[contact_matrix_filter] += Q_full[contact_matrix_filter]
        q[contact_filter] += q_full[contact_filter]

        try:
            impulse_full = pbmm(
                reorder_mat,
                self.solver(  #.apply(
                    J_M,
                    pbmm(reorder_mat.transpose(-1, -2),
                         q).squeeze(-1)).detach().unsqueeze(-1))
                    # pbmm(reorder_mat.transpose(-1, -2), q).squeeze(-1),
                    #eps).detach().unsqueeze(-1))
        except:
            print(f'J_M: {J_M}')
            print(f'reordered q: {pbmm(reorder_mat.transpose(-1, -2), q)}')
            pdb.set_trace()


        impulse = torch.zeros_like(impulse_full)
        impulse[contact_filter] += impulse_full[contact_filter]

        return v_minus + torch.linalg.solve(M, pbmm(J.transpose(-1, -2),
                                                    impulse)).squeeze(-1)

    def sim_step(self, x: Tensor, carry: Tensor) -> Tuple[Tensor, Tensor]:
        """``Integrator.partial_step`` wrapper for
        :py:meth:`forward_dynamics`."""
        q, v = self.space.q_v(x)
        # pylint: disable=E1103
        u = torch.zeros(q.shape[:-1] + (0,))
        v_plus = self.forward_dynamics(q, v, u)
        return v_plus, carry

    def summary(self, statistics: Dict) -> SystemSummary:
        """Generates summary statistics for multibody system.

        The scalars returned are simply the scalar description of the
        system's :py:class:`MultibodyTerms`.

        Meshes are generated for learned
        :py:class:`~dair_pll.geometry.DeepSupportConvex` es.

        Args:
            statistics: Updated evaluation statistics for the model.

        Returns:
            Scalars and meshes packaged into a ``SystemSummary``.
        """
        scalars, meshes = self.multibody_terms.scalars_and_meshes()
        videos = cast(Dict[str, Tuple[np.ndarray, int]], {})

        return SystemSummary(scalars=scalars, videos=videos, meshes=meshes)



class DeepStateAugment3D(Module):
    """To assist with the learning process, replace the quaternion angular
    representation with the rotation matrix vector."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        # Note: The below lines only work because the fixed ground does not
        # contribute to the state of the overall object-ground system.
        quat = x[..., :4]
        rotmat_vec = quaternion_to_rotmat_vec(quat)

        return torch.cat((rotmat_vec, x[..., 4:]), dim=1)

    # TODO:  write compute_jacobian function
