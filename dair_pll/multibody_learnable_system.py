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
from tensordict.tensordict import TensorDict, TensorDictBase
from torch.nn import Module, ParameterList, Parameter
import torch.nn as nn

from dair_pll import urdf_utils, tensor_utils, file_utils
from dair_pll.drake_system import DrakeSystem
from dair_pll.integrator import VelocityIntegrator
from dair_pll.multibody_terms import MultibodyTerms, InertiaLearn
from dair_pll.quaternion import quaternion_to_rotmat_vec
from dair_pll.solvers import DynamicCvxpyLCQPLayer
from dair_pll.state_space import FloatingBaseSpace, StateSpace
from dair_pll.system import System, SystemSummary
from dair_pll.tensor_utils import pbmm, broadcast_lorentz, \
    one_vector_block_diagonal, project_lorentz, reflect_lorentz

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

    def __init__(self,
                 init_urdfs: Dict[str, str],
                 dt: float,
                 w_pred: float,
                 w_comp: float,
                 w_diss: float,
                 w_pen: float,
                 w_res: float,
                 w_res_w: float,
                 w_dev: float,
                 inertia_mode: InertiaLearn = InertiaLearn(),
                 constant_bodies: List[str] = [],
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
            inertia_mode: An InertiaLearn() object specifying which inertial
              parameters to learn
            constant_bodies: list of body names whose properties should NOT
              be learned
            output_urdfs_dir: Optionally, a directory that learned URDFs can be
              written to.
            randomize_initialization: Whether to randomize and export the
              initialization or not.
        """

        multibody_terms = MultibodyTerms(init_urdfs, inertia_mode,
                                         constant_bodies,
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

        self.visualization_system = None
        self.solver = DynamicCvxpyLCQPLayer()
        self.dt = dt
        self.set_carry_sampler(lambda: torch.tensor([False]))
        self.max_batch_dim = 1
        self.w_pred = w_pred
        self.w_comp = w_comp
        self.w_diss = w_diss
        self.w_dev = w_dev
        self.w_pen = w_pen
        self.w_res = w_res
        self.w_res_w = w_res_w

        self.residual_net = None

        self.debug = 0

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
                         contact_forces: Dict[Tuple[str, str], Tensor] = {},
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
            contact_forces: mapping (obj_a_name, obj_b_name) to force on obj_b in World Frame
            loss_pool: optional processing pool to enable multithreaded solves.

        Returns:
            (\*,) loss batch.
        """
        loss_pred, loss_comp, loss_pen, loss_diss, loss_dev = \
            self.calculate_contactnets_loss_terms(x, u, x_plus, contact_forces)

        regularizers = self.get_regularization_terms(x, u, x_plus)

        # For now the regularization terms are: 0) residual norm, 1) residual
        # weights, 2) inertia matrix condition number.  Will need to be updated
        # later if more are added.
        reg_norm = regularizers[0]
        reg_weight = regularizers[1]
        reg_inertia_cond = regularizers[2]

        loss_q_pred = self.space.config_square_error(self.space.euler_step(self.space.q(x), self.space.v(x), self.dt), self.space.q(x_plus))

        loss = (self.w_res * reg_norm) + (self.w_res_w * reg_weight) + \
               (self.w_pred * loss_pred) + (self.w_comp * loss_comp) + \
               (self.w_pen * loss_pen) + (self.w_diss * loss_diss) + \
               (self.w_dev * loss_dev) + \
               (1e-5 * reg_inertia_cond)

        self.debug = self.debug + 1
        return loss

    def get_regularization_terms(self, x: Tensor, u: Tensor,
                                 x_plus: Tensor, **kwargs) -> List[Tensor]:
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

        # Penalize the condition number of the mass matrix.
        q_plus, v_plus = self.space.q_v(x_plus)
        _, M, _, _, _, _, _, _ = self.get_multibody_terms(q_plus, v_plus, u)
        I_BBcm_B = M[..., :3, :3]
        regularizers.append(torch.linalg.cond(I_BBcm_B))

        # TODO: Use the believed geometry to help supervise the learned CoM.
        # if (self.multibody_terms.inertia_mode_txt != 'none') and \
        #    (self.multibody_terms.inertia_mode_txt != 'masses'):
        #     # This means the CoM locations are getting learned.
        #     pass
        return regularizers

    def calculate_contactnets_loss_terms(self,
                         x: Tensor,
                         u: Tensor,
                         x_plus: Tensor,
                         contact_forces: Dict[Tuple[str, str], Tensor] = {}) -> \
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
            (*,) deviation from measurement loss
        """
        # pylint: disable-msg=too-many-locals
        v = self.space.v(x)
        q_plus, v_plus = self.space.q_v(x_plus)
        dt = self.dt
        eps = 1e-8 # TODO: HACK, make a hyperparameter

        # Begin loss calculation.
        delassus, M, J, phi, non_contact_acceleration, obj_pair_list, R_FW_list, mu_list = \
            self.get_multibody_terms(q_plus, v_plus, u)

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

        # Units: Energy
        Q_delassus = delassus + eps * torch.eye(3 * n_contacts) # Force PD

        dv = (v_plus - (v + non_contact_acceleration * dt)).unsqueeze(-2)

        # Constant Terms
        # Calculate the prediction constant based on loss formulation mode.
        constant_pred = 0.5 * pbmm(dv, pbmm(M, dv.transpose(-1, -2)))
        constant_pen = (torch.maximum(
                            -phi, torch.zeros_like(phi))**2).sum(dim=-1)
        constant_pen = constant_pen.reshape(constant_pen.shape + (1,1))


        # Calculate q vectors
        # Final Units: Energy -> q units velocity
        q_pred = -pbmm(J, dv.transpose(-1, -2))
        q_comp = (1.0/dt) * torch.abs(phi_then_zero).unsqueeze(-1)
        q_diss = torch.cat((sliding_speeds, sliding_velocities), dim=-2)
        # Penalize Deviation from measured contact impulses
        # This is in impulse^2, but take deviation w.r.t. Delassus to
        # add 1/mass term to bring into Energy.
        q_dev = torch.zeros_like(q_pred)
        Q_dev = torch.zeros_like(Q_delassus)
        constant_dev = torch.zeros_like(constant_pred)
        for key in contact_forces.keys():
            if key in obj_pair_list:
                idx = obj_pair_list.index(key)
                impulse_measured_W = contact_forces[key].unsqueeze(-1) * dt
                # Constant term is lambda_m magnitude
                constant_dev = constant_dev + 0.5 * pbmm(impulse_measured_W.transpose(-1, -2), impulse_measured_W)
                # q term is lambda_m in contact frame
                impulse_measured_c = pbmm(R_FW_list[idx].transpose(-1, -2), impulse_measured_W)
                # Normal impulse
                q_dev[..., idx, :] = impulse_measured_c[..., 2, :]
                # Scale friction impulse by mu
                q_dev[..., len(obj_pair_list)+2*idx:len(obj_pair_list)+2*(idx+1), :] = impulse_measured_c[..., :2, :] * mu_list[idx]
                # Set 3 diagonal elements (normal, and 2 transverse) to 1 in quadratic term
                Q_dev[..., idx, idx] = 1.0
                # Scale friction terms by mu^2
                for diag_idx in (len(obj_pair_list)+2*idx, (len(obj_pair_list)+2*idx) + 1):
                    Q_dev[..., diag_idx, diag_idx] = 1.0 * mu_list[idx] * mu_list[idx]
        Q_final = Q_delassus + (self.w_dev/self.w_pred)*Q_dev

        q_final = q_pred + (self.w_comp/self.w_pred)*q_comp + \
                     (self.w_diss/self.w_pred)*q_diss + \
                     (self.w_dev/self.w_pred)*q_dev

        # Envelope theorem guarantees that gradient of loss w.r.t. parameters
        # can ignore the gradient of the impulses w.r.t. the QCQP parameters.
        # Therefore, we can detach ``impulses`` from pytorch's computation graph
        # without causing error in the overall loss gradient.
        # pylint: disable=E1103
        try:
            impulses = pbmm(
                reorder_mat,
                self.solver(
                    pbmm(reorder_mat.transpose(-1, -2), pbmm(Q_final, reorder_mat)), # Quadratic Term
                    pbmm(reorder_mat.transpose(-1, -2), q_final).squeeze(-1), # Linear Term
                ).detach().unsqueeze(-1))
        except:
            print(f'reordered Q: {pbmm(reorder_mat.transpose(-1,-2), J_M)}')
            print(f'reordered q: {pbmm(reorder_mat.transpose(-1, -2), q_final)}')
            pdb.set_trace()

        # Hack: remove elements of ``impulses`` where solver likely failed.
        invalid = torch.any((impulses.abs() > 1e3) | impulses.isnan() | impulses.isinf(),
                            dim=-2,
                            keepdim=True)

        constant_pen[invalid] *= 0.
        constant_pred[invalid] *= 0.
        impulses[invalid.expand(impulses.shape)] = 0.

        loss_pred = 0.5 * pbmm(impulses.transpose(-1, -2), pbmm(Q_delassus, impulses)) \
                    + pbmm(impulses.transpose(-1, -2), q_pred) + constant_pred
        loss_comp = pbmm(impulses.transpose(-1, -2), q_comp)
        loss_pen = constant_pen
        loss_diss = pbmm(impulses.transpose(-1, -2), q_diss)
        loss_dev = 0.5 * pbmm(impulses.transpose(-1, -2), pbmm(Q_dev, impulses)) \
                    + pbmm(impulses.transpose(-1, -2), q_dev) + constant_dev

        #if self.debug % 50 == 1:
        #    breakpoint()

        return loss_pred.reshape(-1), loss_comp.reshape(-1), \
               loss_pen.reshape(-1), loss_diss.reshape(-1), \
               loss_dev.reshape(-1)

    def get_multibody_terms(self, q: Tensor, v: Tensor,
        u: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, List, List]:
        """Get multibody terms of the system.  Without a residual, this is a
        straightfoward pass-through to the system's :py:class:`MultibodyTerms`.
        With a residual, the residual augments the continuous dynamics."""

        delassus, M, J, phi, non_contact_acceleration, obj_pair_list, R_FW_list, mu_list = self.multibody_terms(
            q, v, u)

        if self.residual_net != None:
            # Get the residual network's contribution.
            x = torch.cat((q, v), dim=1)
            residual = self.residual_net(x)/self.dt
            amended_acceleration = non_contact_acceleration + residual

        else:
            amended_acceleration = non_contact_acceleration

        return delassus, M, J, phi, amended_acceleration, obj_pair_list, R_FW_list, mu_list

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
        phi_eps = 1e6
        eps = 1e-8 # TODO: HACK make this a hyperparameter
        delassus, M, J, phi, non_contact_acceleration, _, _ = \
            self.get_multibody_terms(q, v, u)
        n_contacts = phi.shape[-1]
        contact_filter = (broadcast_lorentz(phi) <= phi_eps).unsqueeze(-1)
        contact_matrix_filter = pbmm(contact_filter.int(),
                                     contact_filter.transpose(-1,
                                                              -2).int()).bool()

        reorder_mat = tensor_utils.sappy_reorder_mat(n_contacts)
        reorder_mat = reorder_mat.reshape((1,) * (delassus.dim() - 2) +
                                          reorder_mat.shape).expand(
                                              delassus.shape)

        Q_delassus = delassus + eps * torch.eye(3 * n_contacts)

        # pylint: disable=E1103
        double_zero_vector = torch.zeros(phi.shape[:-1] + (2 * n_contacts,))
        phi_then_zero = torch.cat((phi, double_zero_vector),
                                  dim=-1).unsqueeze(-1)

        v_minus = v + dt * non_contact_acceleration
        q_full = pbmm(J, v_minus.unsqueeze(-1)) + (1 / dt) * phi_then_zero

        try:
            impulse_full = pbmm(
                reorder_mat,
                self.solver(
                    pbmm(reorder_mat.transpose(-1, -2), pbmm(Q_delassus, reorder_mat)), # Quadratic Term
                    pbmm(reorder_mat.transpose(-1, -2), q_final).squeeze(-1), # Linear Term
                 ).detach().unsqueeze(-1))
        except:
            print(f'J_M: {J_M}')
            print(f'reordered q: {pbmm(reorder_mat.transpose(-1, -2), q_full)}')
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

    def construct_state_tensor(self,
        data_state: Tensor) -> Tensor:
        """ Input:
            data_state: Tensor coming from the TrajectorySet Dataloader,
                        this class expects a TensorDict, shape [batch, ?]
            Returns: full state tensor (adding traj parameters) shape [batch, n_x_full]
        """

        # TODO: HACK "state" is hard-coded, switch to local arg

        if isinstance(data_state, TensorDictBase):
            return data_state["state"]

        return data_state


class MultibodyLearnableSystemWithTrajectory(MultibodyLearnableSystem):
    """:py:class:`MultibodyLearnableSystem` where a model can have 
        learnable trajectories."""
    model_spaces: Dict[str, StateSpace]
    r"""Map of model name to state space, ignoring spaces where n_x == 0"""
    trajectory_model: str
    r"""Name of the model corresponding to the trajectory"""
    trajectory: ParameterList
    r"""List of parameters length == length of trajectory, each param shape == (1, n_x)"""

    # TODO: Allow multi models to have learnable trajectories


    def __init__(self,
                 trajectory_model: str,
                 traj_len: int,
                 true_traj: Optional[Tensor] = None,
                 **kwargs) -> None:
        ## Construct Super System
        super().__init__(**kwargs)
        self.trajectory_model = trajectory_model

        ## Populate Model Spaces
        self.model_spaces = {}
        plant_diagram = self.multibody_terms.plant_diagram
        for model_id, space in zip(plant_diagram.model_ids, plant_diagram.space.spaces):
            self.model_spaces[plant_diagram.plant.GetModelInstanceName(model_id)] = space

        ## Create Trajectory Parameters
        model_n_x = self.model_spaces[trajectory_model].n_x
        # TODO: HACK set this to all zeros instead of hard-coding
        model_state = torch.vstack([torch.tensor([0.1, 0.00524, 0., 0., 0., 0.])] * traj_len)
        if true_traj is not None:
            model_state = torch.clone(torch.hstack((true_traj["state"].squeeze()[:, :3], true_traj["state"].squeeze()[:, 5:8])))
        self.trajectory = ParameterList([Parameter(model_state[idx, :], requires_grad=True) for idx in range(traj_len)])
        self.trajectory[50].register_hook(lambda grad: print(f"Trajectory Gradient: {grad}"))

    def construct_state_tensor(self,
        data_state: Tensor) -> Tensor:
        """ Input:
            data_state: Tensor coming from the TrajectorySet Dataloader,
                        this class expects a TensorDict, shape [batch, ?]
            Returns: full state tensor (adding traj parameters) shape [batch, n_x_full]
        """

        # Fill Partial States
        assert isinstance(data_state, TensorDictBase)
        model_states = {}
        for model, _ in self.model_spaces.items():
            key = model + "_state"
            if key in data_state.keys():
                model_states[model] = data_state[key]
                if len(model_states[model].shape) == 1:
                    model_states[model] = model_states[model].reshape(model_states.shape[0], 1)

        # Input Sanitation
        assert data_state["time"].shape == data_state.shape + (1,)
        for model, state in model_states.items():
            assert state.shape == data_state.shape + (self.model_spaces[model].n_x,)
        
        # Get trajectory parameters
        test = [self.trajectory[int(i)] for i in data_state["time"].flatten()]
        traj_x = torch.stack(test)
        traj_x = traj_x.reshape(data_state.shape + (self.model_spaces[self.trajectory_model].n_x,)) # [batch x traj_n_x]

        # Loop through models and construct state
        ret_q = torch.tensor([])
        ret_v = torch.tensor([])
        for model, space in self.model_spaces.items():
            # Ignore world and other degenerate spaces
            if space.n_x == 0:
                continue

            # Select model state or trajectory state
            model_x = traj_x
            if model != self.trajectory_model:
                model_x = model_states[model]

            # Append to return value
            if ret_q.numel() == 0:
                ret_q = model_x[..., :space.n_q]
                ret_v = model_x[..., space.n_q:]
            else:
                ret_q = torch.cat((ret_q, model_x[..., :space.n_q]), dim=-1)
                ret_v = torch.cat((ret_v, model_x[..., space.n_q:]), dim=-1)

        # Return full state batch
        return torch.cat((ret_q, ret_v), dim=-1)

    def contactnets_loss(self,
                         x: Tensor,
                         u: Tensor,
                         x_plus: Tensor,
                         contact_forces: Dict[Tuple[str, str], Tensor] = {},
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
            contact_forces: mapping (obj_a_name, obj_b_name) to force on obj_b in World Frame
            loss_pool: optional processing pool to enable multithreaded solves.

        Returns:
            (\*,) loss batch.
        """
        loss = super().contactnets_loss(x, u, x_plus, contact_forces, loss_pool)

        # Add Prediction term for v->q, not covered by dynamics prediction term
        # TODO: HACK add q_pred weight to config
        loss_q_pred = self.space.config_square_error(self.space.euler_step(self.space.q(x), self.space.v(x), self.dt), self.space.q(x_plus))

        loss += 1e2 * loss_q_pred
        return loss


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
