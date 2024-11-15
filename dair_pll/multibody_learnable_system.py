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

from os import path
import pdb
from typing import Any, List, Tuple, Optional, Dict, cast, Union

import gin
import numpy as np
import torch
from torch import Tensor
from tensordict.tensordict import TensorDictBase

from dair_pll import urdf_utils, tensor_utils, file_utils
from dair_pll.drake_system import DrakeSystem
from dair_pll.integrator import VelocityIntegrator
from dair_pll.learnable_trajectory import LearnableTrajectories
from dair_pll.multibody_terms import MultibodyTerms, LearnableBodySettings
from dair_pll.solvers import DynamicCvxpyLCQPLayer
from dair_pll.state_space import StateSpace, ProductSpace
from dair_pll.system import SystemSummary
from dair_pll.tensor_utils import pbmm, broadcast_lorentz

# Scaling factors to equalize translation and rotation errors.
# For rotation versus linear scaling:  penalize 0.1 meters same as 90 degrees.
ROTATION_SCALING = 0.2 / torch.pi
# For articulation versus linear/rotation scaling:  penalize the scenario where
# one elbow link is in the right place and the other is 180 degrees flipped the
# same, whether link 1 or link 2 are in the right place.
ELBOW_COM_TO_AXIS_DISTANCE = 0.035
JOINT_SCALING = 2 * ELBOW_COM_TO_AXIS_DISTANCE / torch.pi + ROTATION_SCALING

# Dimension of Measured Force
DIMENSION = 3


@gin.configurable
class MultibodyLearnableSystem(DrakeSystem):
    """:py:class:`System` interface for dynamics associated with
    :py:class:`MultibodyTerms`."""

    multibody_terms: MultibodyTerms
    init_urdfs: Dict[str, str]
    output_urdfs_dir: Optional[str] = None
    visualization_system: Optional[DrakeSystem]
    solver: DynamicCvxpyLCQPLayer
    dt: float
    loss_cache: Dict[str, Any]

    def __init__(
        self,
        init_urdfs: Dict[str, str],
        dt: float,
        w_pred: float,
        w_q_pred: float,
        w_comp: float,
        w_diss: float,
        w_pen: float,
        w_dev: float,
        w_reg_iner: float,
        learnable_body_dict: Optional[Dict[str, LearnableBodySettings]] = None,
        output_urdfs_dir: Optional[str] = None,
        represent_geometry_as: str = "box",
        randomize_initialization: bool = False,
    ) -> None:
        """Inits :py:class:`MultibodyLearnableSystem` with provided model URDFs.

        Implementation is primarily based on Drake. Bodies are modeled via
        :py:class:`MultibodyTerms`, which uses Drake symbolics to generate
        dynamics terms, and the system can be exported back to a
        Drake-interpretable representation as a set of URDFs.

        Args:
            init_urdfs: Names and corresponding URDFs to model with
              :py:class:`MultibodyTerms`.
            dt: Time step of system in seconds.
            learnable_body_dict: dict of body names and which properties should
              be learned
            output_urdfs_dir: Optionally, a directory that learned URDFs can be
              written to.
            randomize_initialization: Whether to randomize and export the
              initialization or not.
        """
        if learnable_body_dict is None:
            learnable_body_dict = {}

        multibody_terms = MultibodyTerms(
            init_urdfs,
            learnable_body_dict,
            represent_geometry_as,
        )

        space = multibody_terms.plant_diagram.space
        integrator = VelocityIntegrator(space, self.sim_step, dt)
        super(DrakeSystem, self).__init__(space, integrator)

        self.output_urdfs_dir = output_urdfs_dir
        self.multibody_terms = multibody_terms
        self.init_urdfs = init_urdfs

        if randomize_initialization:
            # Add noise and export.
            raise NotImplementedError("Random Initialization Not Implemented")

        self.visualization_system = None
        self.solver = DynamicCvxpyLCQPLayer()
        self.dt = dt
        self.set_carry_sampler(lambda: torch.tensor([False]))
        self.max_batch_dim = 1
        self.w_pred = w_pred
        self.w_q_pred = w_q_pred
        self.w_comp = w_comp
        self.w_diss = w_diss
        self.w_dev = w_dev
        self.w_pen = w_pen
        self.w_reg_iner = w_reg_iner

        # Match DrakeSystem Attributes
        self.urdfs = self.init_urdfs
        self.plant_diagram = multibody_terms.plant_diagram

        self.debug = False
        self.loss_cache = {}

    def set_debug(self, debug: bool = True):
        self.debug = debug

    def generate_updated_urdfs(self, suffix: str = None) -> Dict[str, str]:
        """Exports current parameterization as a :py:class:`DrakeSystem`.

        Returns:
            New Drake system instantiated on new URDFs.
        """
        assert self.output_urdfs_dir is not None
        new_urdf_strings = urdf_utils.represent_multibody_terms_as_urdfs(
            self.multibody_terms, self.output_urdfs_dir
        )

        # saves new urdfs with model name plus optional suffix in
        # new folder.
        for urdf_name, new_urdf_string in new_urdf_strings.items():
            new_urdf_filename = urdf_name + ".urdf"
            if suffix is not None:
                new_urdf_filename = (
                    new_urdf_filename.split(".")[0] + "_" + suffix + ".urdf"
                )

            new_urdf_path = path.join(self.output_urdfs_dir, new_urdf_filename)
            file_utils.save_string(new_urdf_path, new_urdf_string)

        self.urdfs = new_urdf_strings
        return new_urdf_strings

    def contactnets_loss(
        self,
        x: Tensor,
        u: Tensor,
        x_plus: Tensor,
        contact_forces: Optional[Dict[Tuple[str, str], Tensor]] = None,
    ) -> Tensor:
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

        Returns:
            (\*,) loss batch.
        """
        if contact_forces is None:
            contact_forces = {}

        loss_pred, loss_q_pred, loss_comp, loss_pen, loss_diss, loss_dev = (
            self.calculate_contactnets_loss_terms(x, u, x_plus, contact_forces)
        )

        regularizers = self.get_regularization_terms(x, u, x_plus)

        # For now the regularization terms are: 0) inertia matrix condition number.
        # Will need to be updated later if more are added.
        reg_inertia_cond = regularizers[0]

        loss = (
            (self.w_pred * loss_pred)
            + (self.w_q_pred * loss_q_pred)
            + (self.w_comp * loss_comp)
            + (self.w_pen * loss_pen)
            + (self.w_diss * loss_diss)
            + (self.w_dev * loss_dev)
            # TODO: HACK re-add later
#            + (self.w_reg_iner * reg_inertia_cond)
        )

        # Cache Losses
        self.loss_cache["loss_pred"] = loss_pred.clone().detach()
        self.loss_cache["loss_q_pred"] = loss_q_pred.clone().detach()
        self.loss_cache["loss_comp"] = loss_comp.clone().detach()
        self.loss_cache["loss_pen"] = loss_pen.clone().detach()
        self.loss_cache["loss_diss"] = loss_diss.clone().detach()
        self.loss_cache["loss_dev"] = loss_dev.clone().detach()

        return loss

    def get_regularization_terms(
        self, x: Tensor, u: Tensor, x_plus: Tensor, **kwargs
    ) -> List[Tensor]:
        """Calculate some regularization terms."""

        regularizers = []

        # Penalize the condition number of the mass matrix.
        q_plus, v_plus = self.space.q_v(x_plus)
        _, M, _, _, _, _, _, _ = self.get_multibody_terms(q_plus, v_plus, u)
        # TODO HACK: hard-coded. rows/cols in M should match  model_body_qw in GetStateNames()
        I_BBcm_B = M[..., 3:6, 3:6]
        # pylint doesn't know about torch functions
        # pylint: disable=E1102
        regularizers.append(torch.linalg.cond(I_BBcm_B))

        # TODO: Use the believed geometry to help supervise the learned CoM.
        return regularizers

    def calculate_contactnets_loss_terms(
        self,
        x: Tensor,
        u: Tensor,
        x_plus: Tensor,
        contact_forces: Optional[Dict[Tuple[str, str], Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
            (*,) prediction (q_plus = q + v) error loss
            (*,) complementarity violation loss.
            (*,) penetration loss.
            (*,) dissipation violation loss.
            (*,) deviation from measurement loss
        """
        if contact_forces is None:
            contact_forces = {}

        v = self.space.v(x)
        q_plus, v_plus = self.space.q_v(x_plus)
        dt = self.dt
        eps = 1e-8  # TODO: HACK, make a hyperparameter

        # Begin loss calculation.
        (
            delassus,
            M,
            J,
            phi,
            non_contact_acceleration,
            obj_pair_list,
            R_FW_list,
            mu_list,
        ) = self.get_multibody_terms(q_plus, v_plus, u, contact_forces)

        # Construct a reordering matrix s.t. lambda_CN = reorder_mat @ f_sappy.
        n_contacts = phi.shape[-1]
        reorder_mat = tensor_utils.sappy_reorder_mat(n_contacts)
        reorder_mat = reorder_mat.reshape(
            (1,) * (delassus.dim() - 2) + reorder_mat.shape
        ).expand(delassus.shape)

        double_zero_vector = torch.zeros(phi.shape[:-1] + (2 * n_contacts,))
        phi_then_zero = torch.cat((phi, double_zero_vector), dim=-1)

        J_t = J[..., n_contacts:, :]
        sliding_velocities = pbmm(J_t, v_plus.unsqueeze(-1))
        sliding_speeds = sliding_velocities.reshape(
            phi.shape[:-1] + (n_contacts, 2)
        ).norm(dim=-1, keepdim=True)

        J_n = J[..., :n_contacts, :]
        normal_velocities = pbmm(J_n, v_plus.unsqueeze(-1))
        normal_velocities = torch.maximum(normal_velocities, torch.zeros_like(normal_velocities))

        # Units: Energy
        Q_delassus = delassus + eps * torch.eye(3 * n_contacts)  # Force PD

        dv = (v_plus - (v + non_contact_acceleration * dt)).unsqueeze(-2)

        # Constant Terms
        # Calculate the prediction constant based on loss formulation mode.
        constant_pred = 0.5 * pbmm(dv, pbmm(M, dv.transpose(-1, -2)))
        constant_pen = (torch.maximum(-phi, torch.zeros_like(phi)) ** 2).sum(dim=-1)
        constant_pen = constant_pen.reshape(constant_pen.shape + (1, 1))

        # Calculate q vectors
        # Final Units: Energy -> q units velocity
        q_pred = -pbmm(J, dv.transpose(-1, -2))
        q_comp = (1.0 / dt) * torch.maximum(phi_then_zero, torch.zeros_like(phi_then_zero)).unsqueeze(-1)
        q_diss = torch.cat((sliding_speeds, sliding_velocities), dim=-2)
        q_n_diss = torch.cat((normal_velocities, double_zero_vector.unsqueeze(-1)), dim=-2)

        # Penalize Deviation from measured contact impulses
        # This is in impulse^2, but take deviation w.r.t. Delassus to
        # add 1/mass term to bring into Energy.
        q_dev = torch.zeros_like(q_pred)
        Q_dev = torch.zeros_like(Q_delassus)
        constant_dev = torch.zeros_like(constant_pred)

        for key in contact_forces.keys():
            indices = np.array([i for i, x in enumerate(obj_pair_list) if x == key])
            if len(indices) == 0:
                continue
            mu_i = mu_list[indices[0]]
            # Q_dev = diag(mu)RS^TSR^Tdiag(mu)^T; diag(mu) = 1 if normal, mu otherwise
            # R is block diagonal rotation matrices, S is summation matrix
            diag_F_mu = torch.zeros(
                q_dev.shape[:-1] + ((len(indices) * 3),)
            )  # (batch x (n_c_tot*3) x (n_c_obj*3))
            R_FW_mat = torch.zeros(
                q_dev.shape[:-2] + ((len(indices) * 3), (len(indices) * 3))
            )  # (batch x (n_c_obj*3) x (n_c_obj*3))
            sum_W_mat = torch.zeros(
                q_dev.shape[:-2] + (3, (len(indices) * 3))
            )  # (batch x 3 x (n_c_obj*3))
            for contact, idx in enumerate(indices):

                # Map Normal Force
                diag_F_mu[..., idx, contact * 3 + 2] = 1.0
                # Map Tangent Forces
                diag_F_mu[..., len(obj_pair_list) + 2 * idx, contact * 3] = mu_i
                diag_F_mu[..., len(obj_pair_list) + 2 * idx + 1, contact * 3 + 1] = mu_i

                # Create Block diagonal matrix (note torch.block_diag isn't vectorized)
                R_FW_mat[
                    ...,
                    contact * 3 : (contact + 1) * 3,
                    contact * 3 : (contact + 1) * 3,
                ] = R_FW_list[idx]

                # Summation
                sum_W_mat[..., 0, contact * 3] = 1.0
                sum_W_mat[..., 1, contact * 3 + 1] = 1.0
                sum_W_mat[..., 2, contact * 3 + 2] = 1.0
            q_dev_part = pbmm(
                sum_W_mat, pbmm(R_FW_mat.transpose(-1, -2), diag_F_mu.transpose(-1, -2))
            )  # (batch, 3, (n_c_tot*3))
            Q_dev += pbmm(q_dev_part.transpose(-1, -2), q_dev_part)

            # Linear Term is lambda_mSR^Tdiag(mu)^T
            impulse_measured_W = contact_forces[key].unsqueeze(-2) * dt  # (batch, 1, 3)
            q_dev -= pbmm(impulse_measured_W, q_dev_part).transpose(
                -1, -2
            )  # (batch, n_c_tot*3, 1)

            # Constant term is lambda_m magnitude, multiply by 0.5 here to match constant_pred
            constant_dev += 0.5 * pbmm(
                impulse_measured_W, impulse_measured_W.transpose(-1, -2)
            )

        Q_final = Q_delassus + (self.w_dev / self.w_pred) * Q_dev

        q_final = (
            q_pred
            + (self.w_comp / self.w_pred) * q_comp
            + (self.w_diss / self.w_pred) * q_diss
            + (self.w_diss / self.w_pred) * q_n_diss
            + (self.w_dev / self.w_pred) * q_dev
        )

        # Envelope theorem guarantees that gradient of loss w.r.t. parameters
        # can ignore the gradient of the impulses w.r.t. the QCQP parameters.
        # Therefore, we can detach ``impulses`` from pytorch's computation graph
        # without causing error in the overall loss gradient.
        with torch.no_grad():
            impulses = pbmm(
                reorder_mat,
                self.solver(
                    pbmm(
                        reorder_mat.transpose(-1, -2), pbmm(Q_final, reorder_mat)
                    ),  # Quadratic Term
                    pbmm(reorder_mat.transpose(-1, -2), q_final).squeeze(-1),  # Linear Term
                )
                .unsqueeze(-1),
            )

        # Hack: remove elements of ``impulses`` where solver likely failed.
        invalid = torch.any(
            (impulses.abs() > 1e3) | impulses.isnan() | impulses.isinf(),
            dim=-2,
            keepdim=True,
        )

        constant_pen[invalid] *= 0.0
        constant_pred[invalid] *= 0.0
        constant_dev[invalid] *= 0.0
        impulses[invalid.expand(impulses.shape)] = 0.0

        loss_pred = (
            0.5 * pbmm(impulses.transpose(-1, -2), pbmm(Q_delassus, impulses))
            + pbmm(impulses.transpose(-1, -2), q_pred)
            + constant_pred
        )
        loss_q_pred = self.space.config_square_error(
            self.space.euler_step(self.space.q(x), self.space.v(x), self.dt),
            self.space.q(x_plus),
        )
        loss_comp = pbmm(impulses.transpose(-1, -2), q_comp)
        loss_pen = constant_pen
        loss_diss = pbmm(impulses.transpose(-1, -2), q_diss) + pbmm(impulses.transpose(-1, -2), q_n_diss)
        loss_dev = (
            0.5 * pbmm(impulses.transpose(-1, -2), pbmm(Q_dev, impulses))
            + pbmm(impulses.transpose(-1, -2), q_dev)
            + constant_dev
        )

        if self.debug:
            # pylint: disable-next=forgotten-debug-statement
            pdb.Pdb(nosigint=True).set_trace()

        # Check for positive definite deviation loss
        try:
            assert np.all(loss_dev.detach().cpu().numpy() > 0.0)
        except AssertionError:
            # pylint: disable-next=forgotten-debug-statement
            pdb.Pdb(nosigint=True).set_trace()

        return (
            loss_pred.reshape(-1),
            loss_q_pred.reshape(-1),
            loss_comp.reshape(-1),
            loss_pen.reshape(-1),
            loss_diss.reshape(-1),
            loss_dev.reshape(-1),
        )

    def get_multibody_terms(
        self,
        q: Tensor,
        v: Tensor,
        u: Tensor,
        estimated_normals_W: Optional[Dict[Tuple[str, str], Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, List, List]:
        """Get multibody terms of the system.  Without a residual, this is a
        straightfoward pass-through to the system's :py:class:`MultibodyTerms`.
        With a residual, the residual augments the continuous dynamics."""

        if estimated_normals_W is None:
            estimated_normals_W = {}

        (
            delassus,
            M,
            J,
            phi,
            non_contact_acceleration,
            obj_pair_list,
            R_FW_list,
            mu_list,
        ) = self.multibody_terms(q, v, u, estimated_normals_W)

        return (
            delassus,
            M,
            J,
            phi,
            non_contact_acceleration,
            obj_pair_list,
            R_FW_list,
            mu_list,
        )

    def forward_dynamics(self, q: Tensor, v: Tensor, u: Tensor) -> Tensor:
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

        Returns:
            (\*, space.n_v) delta velocity batch.
        """
        # pylint: disable=too-many-locals
        dt = self.dt
        phi_eps = 1e6
        eps = 1e-8  # TODO: HACK make this a hyperparameter
        delassus, M, J, phi, non_contact_acceleration, _, _ = self.get_multibody_terms(
            q, v, u
        )
        n_contacts = phi.shape[-1]
        contact_filter = (broadcast_lorentz(phi) <= phi_eps).unsqueeze(-1)

        reorder_mat = tensor_utils.sappy_reorder_mat(n_contacts)
        reorder_mat = reorder_mat.reshape(
            (1,) * (delassus.dim() - 2) + reorder_mat.shape
        ).expand(delassus.shape)

        Q_delassus = delassus + eps * torch.eye(3 * n_contacts)

        # pylint: disable=E1103
        double_zero_vector = torch.zeros(phi.shape[:-1] + (2 * n_contacts,))
        phi_then_zero = torch.cat((phi, double_zero_vector), dim=-1).unsqueeze(-1)

        v_minus = v + dt * non_contact_acceleration
        q_full = pbmm(J, v_minus.unsqueeze(-1)) + (1 / dt) * phi_then_zero

        with torch.no_grad():
            impulse_full = pbmm(
                reorder_mat,
                self.solver(
                    pbmm(
                        reorder_mat.transpose(-1, -2), pbmm(Q_delassus, reorder_mat)
                    ),  # Quadratic Term
                    pbmm(reorder_mat.transpose(-1, -2), q_full).squeeze(-1),  # Linear Term
                )
                .unsqueeze(-1),
            )

        impulse = torch.zeros_like(impulse_full)
        impulse[contact_filter] += impulse_full[contact_filter]

        # pylint doesn't know about torch functions
        # pylint: disable=E1102
        return v_minus + torch.linalg.solve(
            M, pbmm(J.transpose(-1, -2), impulse)
        ).squeeze(-1)


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


@gin.configurable("LearnableSystem")
class MultibodyLearnableSystemWithTrajectory(MultibodyLearnableSystem):
    """:py:class:`MultibodyLearnableSystem` where a model can have
    learnable trajectories."""

    _model_spaces: Dict[str, StateSpace]
    r"""Map of model name to state space, ignoring spaces where n_x == 0"""
    _trajectory_model_names: List[str]
    r"""Name of the model corresponding to the trajectory"""
    _trajectory: LearnableTrajectories
    r"""The learnable trajectory for the given models"""

    def __init__(
        self,
        trajectory_model_names: Union[List[str], str],
        init_traj_state: Optional[Union[List[float], Tensor]] = None,
        **kwargs,
    ) -> None:
        ## Construct Super System
        super().__init__(**kwargs)
        self._trajectory_model_names = (
            trajectory_model_names
            if isinstance(trajectory_model_names, list)
            else [trajectory_model_names]
        )

        ## Populate Model Spaces
        self.model_spaces = {}
        traj_spaces = []
        plant_diagram = self.multibody_terms.plant_diagram
        for model_id, space in zip(plant_diagram.model_ids, plant_diagram.space.spaces):
            name = plant_diagram.plant.GetModelInstanceName(model_id)
            self.model_spaces[name] = space
            if name in self._trajectory_model_names:
                traj_spaces.append(space)

        ## Create Trajectory Parameters
        init_state = None
        if init_traj_state is not None:
            init_state = (
                init_traj_state
                if isinstance(init_traj_state, Tensor)
                else torch.tensor(init_traj_state)
            )
        self._trajectory = LearnableTrajectories(ProductSpace(traj_spaces), init_state)

    def add_trajectories(
        self, traj_lens: List[int], traj_data: Optional[List[Optional[Tensor]]] = None
    ):
        """Add new learnable trajectory of length"""
        if traj_data is None:
            traj_data = [None] * len(traj_lens)
        assert len(traj_data) == len(traj_lens)
        for traj_len, traj_datum in zip(traj_lens, traj_data):
            self._trajectory.add_trajectory(traj_len, traj_datum)

    def construct_state_tensor(
        self, data_state: Tensor, state_key: Optional[str] = None
    ) -> Tensor:
        """Input:
        data_state: Tensor coming from the TrajectorySet Dataloader,
                    this class expects a TensorDict, shape [batch, ?]
        state_key: if set, override return with this key
        Returns: full state tensor (adding traj parameters) shape [batch, n_x_full]
        """
        if state_key is not None and state_key in data_state:
            return data_state[state_key]

        # Don't mutate input
        assert isinstance(data_state, TensorDictBase)
        data_state = data_state.clone()

        # Inject trajectory_model's state if not already present
        if "traj_num" in data_state and "index" in data_state:
            assert data_state["traj_num"].numel() == data_state.numel()
            assert data_state["index"].numel() == data_state.numel()
            traj_states = self._trajectory(data_state["traj_num"].squeeze(-1), data_state["index"].squeeze(-1))
            traj_splits = self._trajectory.space.x_split(traj_states)
            for traj_model_idx, traj_model_name in enumerate(
                self._trajectory_model_names
            ):
                if (traj_model_name + "_state") not in data_state:
                    model_x = traj_splits[traj_model_idx]
                    data_state[traj_model_name + "_state"] = model_x

        # Return full state using DrakeSystem's function
        return super().construct_state_tensor(data_state)
