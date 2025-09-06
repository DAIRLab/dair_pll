"""Construction and analysis of learnable multibody systems.

Similar to Drake, multibody systems are instantiated as a child class of
:py:class:`System`: :py:class:`MultibodyTactileLearnableSystem`. This object is a thin
wrapper for a :py:class:`MultibodyTerms` member variable, which manages
computation of lumped terms necessary for simulation and evaluation.

'Tactile' means that the system is supervised with contact forces / normals directly.
See whitepaper for loss terms: https://www.overleaf.com/project/682b4c612359b4208bf974dd

Simulation is implemented via Anitescu's [1] convex method.

A large portion of the internal implementation of :py:class:`DrakeSystem` is
implemented in :py:class:`MultibodyPlantDiagram`.

[1] M. Anitescu, “Optimization-based simulation of nonsmooth rigid
multibody dynamics,” Mathematical Programming, 2006,
https://doi.org/10.1007/s10107-005-0590-7
"""

# pylint: disable=too-many-lines

from dataclasses import dataclass, field
from enum import Enum
from functools import partial
import math
import time
from typing import override, Optional, cast, Union

import diffcp
import gin
import numpy as np
import pydrake
from pydrake.geometry import Shape
from scipy.spatial.transform import Rotation
from tensordict import TensorDict
import torch
from torch import Tensor
from torch.nn import Module, Parameter

from dair_pll.dataset_management import TrajectorySet
from dair_pll.drake_utils import (
    unique_body_identifier,
    get_bodies_in_model_instance,
)
from dair_pll.geometry import (
    CollisionGeometry,
    PydrakeToCollisionGeometryFactory,
    _NOMINAL_HALF_LENGTH,
)
from dair_pll.learnable_trajectory import LearnableTrajectories
from dair_pll.multibody_terms import MultibodyTerms, LearnableBodySettings
from dair_pll.solvers import jaxopt_solver, DynamicCvxpyLCQPLayer
from dair_pll.state_space import StateSpace, ProductSpace
from dair_pll.tensor_utils import pbmm, broadcast_lorentz, sappy_reorder_mat, stable_inv
from dair_pll.quaternion import quaternion_to_rotmat_vec


@gin.constants_from_enum
class ModelType(Enum):
    """Type for each URDF model"""

    NONE = 0
    CONTROLLABLE = 1
    LEARNABLE = 2


@gin.constants_from_enum
class LossFunction(Enum):
    """Which Loss Function to Use"""

    NIMP = 0
    VIMP = 1


@gin.configurable("TactileHyperparameters")
@dataclass
class MultibodyTactileHyperparameters:
    """Class to specify hyperparameters"""

    # pylint: disable=too-many-instance-attributes

    default_dt: float = 1.0 / 30.0  # 30 Hz Default
    dt_thresh: float = 1.0  # When to assume a trajectory boundary
    ctrl_kp: float = 20.0  # KP for PD Controller
    ctrl_kd: float = 10.0  # KD for PD Controller
    rsim_eps: float = 1e0  # eps for reverse simulation Jacobian inverse

    # Switches
    loss_fn: LossFunction = field(default_factory=lambda: LossFunction.NIMP)
    supervise_non_contact_force: bool = False

    # Loss Weights
    w_phi_nominal: float = 0.002  # m, distance where p(contact_measured) drops below CI
    w_phi_ci: float = 0.05  # Confidence Interval [0,1] for above
    w_normal_var: float = (
        0.01519224261  # cos(radians) [default 10 degrees], variance of cos(normal angle deviation)
    )
    w_force_var: float = (
        1e-2  # N, variance of contact force measurement (assume identity covariance)
    )
    w_pen: float = 1e0  # cost/m
    w_v_pred: float = 1e0  # cost/J
    w_q_pred: float = 1e0  # cost/J
    w_comp: float = 1e0  # cost/J
    w_diss: float = 1e0  # cost/J
    w_elas: float = 1e0  # cost/J


@gin.configurable("TactileSystem")
class MultibodyLearnableTactileSystem(Module):
    """:py:class:`System` interface for dynamics associated with
    :py:class:`MultibodyTerms` and Tactile Supervision."""

    _multibody_terms: MultibodyTerms
    r"""Underlying drake multibody computations object"""
    _learned_model_names: list[str]
    r"""All controlled models in-order."""
    _learned_trajectory: LearnableTrajectories
    r"""The learnable trajectory for the given models"""
    _controlled_model_names: list[str]
    r"""All controlled models in-order."""
    _controlled_space: StateSpace
    r"""The StateSpace that can be controlled."""

    _solver: DynamicCvxpyLCQPLayer
    r"""SOC QP Solver"""
    _hyperparameters: MultibodyTactileHyperparameters
    r"""Hyperparameter object"""

    def __init__(
        self,
        urdfs: dict[ModelType, dict[str, str]],
        learnable_body_dict: Optional[dict[str, LearnableBodySettings]] = None,
        hyperparameters: MultibodyTactileHyperparameters = MultibodyTactileHyperparameters(),
        init_learned_state: Optional[list[float] | Tensor] = None,
    ) -> None:
        """Inits :py:class:`MultibodyLearnableTactileSystem` with provided model URDFs.

        Implementation is primarily based on Drake. Bodies are modeled via
        :py:class:`MultibodyTerms`, which uses Drake symbolics to generate
        dynamics terms, and the system can be exported back to a
        Drake-interpretable representation as a set of URDFs.

        Args:
            urdfs: For each model type: names and corresponding URDFs to model with
              :py:class:`MultibodyTerms`.
            learnable_body_dict: dict of body names and which properties should
              be learned
            hyperparameters: Learning hyperparameters
            init_learned_state: initial state for the learnable models, (learn_spaces.n_x,)
        """
        super().__init__()

        # Input Validation
        if learnable_body_dict is None:
            learnable_body_dict = {}
        assert (
            0.0 < hyperparameters.w_phi_ci < 1.0
        ), f"w_phi_ci must be in (0,1), {hyperparameters.w_phi_ci}"
        assert (
            hyperparameters.w_phi_nominal > 0.0
        ), f"w_phi_nominal must be >0, {hyperparameters.w_phi_nominal}"

        # Init Multibody Terms
        urdf_dict = {}
        for val in urdfs.values():
            urdf_dict.update(val)
        multibody_terms = MultibodyTerms(
            urdf_dict,
            learnable_body_dict,
        )
        self._multibody_terms = multibody_terms

        # TODO: HACK re-add random initialization

        # Fill other class attributes
        # Pylint doesn't know about gin
        # pylint: disable=no-value-for-parameter
        self._solver = jaxopt_solver
        self._solver_learn = DynamicCvxpyLCQPLayer()
        self._hyperparameters = hyperparameters

        ## Populate Model Spaces
        self._learned_model_names = []
        self._controlled_model_names = []
        learn_spaces = []
        ctrl_spaces = []
        for model_id, space in zip(
            multibody_terms.plant_diagram.model_ids,
            multibody_terms.plant_diagram.space.spaces,
        ):
            name = multibody_terms.plant_diagram.plant.GetModelInstanceName(model_id)
            if name in urdfs[ModelType.LEARNABLE]:
                self._learned_model_names.append(name)
                learn_spaces.append(space)
            elif name in urdfs[ModelType.CONTROLLABLE]:
                self._controlled_model_names.append(name)
                ctrl_spaces.append(space)
        self._controlled_space = ProductSpace(ctrl_spaces)
        # For PD control robot needs n_q == n_v
        assert self._controlled_space.n_q == self._controlled_space.n_v

        ## Create Learnable Trajectory
        init_state = None
        if init_learned_state is not None:
            init_state = (
                init_learned_state
                if isinstance(init_learned_state, Tensor)
                else torch.tensor(init_learned_state)
            )
        self._learned_trajectory = LearnableTrajectories(
            ProductSpace(learn_spaces), init_state
        )

    @property
    def space(self):
        """Space of full system"""
        return self._multibody_terms.plant_diagram.space

    @property
    def plant(self):
        """Full system plant"""
        return self._multibody_terms.plant_diagram.plant

    @property
    def controlled_space(self):
        """Controlled / Robot StateSpace"""
        return self._controlled_space

    @property
    def controlled_model_names(self):
        """Controlled / Robot Model Names in Plant"""
        return self._controlled_model_names

    @property
    def learned_model_names(self):
        """Learned Model Names in Plant"""
        return self._learned_model_names

    def add_learnable_trajectories(
        self, traj_lens: list[int], traj_data: Optional[list[Optional[Tensor]]] = None
    ):
        """Extend learnable trajectory by length sum(traj_lens) with data in traj_data"""
        if self._hyperparameters.loss_fn == LossFunction.NIMP:
            print("Warning: Not adding trajectory as x_0 is the only param.")
            return
        if traj_data is None:
            traj_data = [None] * len(traj_lens)
        assert len(traj_data) == len(traj_lens)
        for traj_len, traj_datum in zip(traj_lens, traj_data):
            self._learned_trajectory.add_trajectory(traj_len, traj_datum)

    def forward_dynamics_functional(
        self,
        step_q: Tensor,
        step_v: Tensor,
        step_u: Tensor,
        step_dt: Tensor,
        multibody_params: dict[str, Tensor],
        all_phis: bool = False,
        no_sim: bool = False,
    ) -> Tensor:
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
            step_q: (\*, space.n_q) current configuration batch.
            step_v: (\*, space.n_v) current velocity batch.
            step_u: (\*, ?) current control batch.
            step_dt: (\*, 1) delta t

        Returns:
            (\*, space.n_v) delta velocity batch.
        """
        # pylint: disable=too-many-locals

        # Input Validation
        batch_dims = step_q.size()[:-1]
        assert step_q.size() == batch_dims + (self.space.n_q,)
        assert step_v.size() == batch_dims + (self.space.n_v,)
        assert step_u.size() == batch_dims + (self._controlled_space.n_v,)
        dt_prev = self._hyperparameters.default_dt * torch.ones(batch_dims + (1,))
        if step_dt is not None:
            step_dt_expanded = step_dt.expand(batch_dims + (1,))
            dt_mask = (step_dt_expanded < self._hyperparameters.dt_thresh).float()
            dt = (step_dt_expanded * dt_mask) + dt_prev * (1.0 - dt_mask)
            # Vmap doesn't support boolean mask
            # dt[step_dt_expanded < self._hyperparameters.dt_thresh] = step_dt_expanded[
            #    step_dt_expanded < self._hyperparameters.dt_thresh
            # ]
        else:
            dt = dt_prev

        # TODO: Can safely return 0s if above threshold
        # if dt > self._hyperparameters.dt_thresh:
        #    return (
        #        torch.zeros_like(step_v),
        #        ret_contact_forces,
        #        ret_contact_normals,
        #        ret_phis,
        #    )

        assert dt.size() == batch_dims + (1,)
        phi_eps = 1e-2
        eps = 1e-8
        (
            m_delassus,
            m_mass,
            m_jac,
            m_phi,
            non_contact_acceleration,
            obj_pair_list,
            mr_fw_list,
            mu_list,
        ) = torch.func.functional_call(
            self._multibody_terms, multibody_params, args=(step_q, step_v, step_u)
        )
        n_contacts = m_phi.shape[-1]
        contact_filter = (broadcast_lorentz(m_phi) <= phi_eps).unsqueeze(-1).float()

        mq_delassus = m_delassus + eps * torch.eye(3 * n_contacts)

        # pylint: disable=E1103
        double_zero_vector = torch.zeros(m_phi.shape[:-1] + (2 * n_contacts,))
        # If no contact, phi == 1 without derivative, else unchanged
        phi_then_zero = (
            torch.cat((m_phi, double_zero_vector), dim=-1).unsqueeze(-1)
            * contact_filter
        ) + (1.0 - contact_filter)

        step_v_minus = step_v + dt * non_contact_acceleration
        q_full = (
            pbmm(m_jac, step_v_minus.unsqueeze(-1))
            + torch.reciprocal(dt).unsqueeze(-1) * phi_then_zero
        )

        ### Construct contact forces / normals
        ret_contact_forces = {}  # Dict[Tuple[str, str], Tensor]
        ret_contact_normals = {}  # Dict[Tuple[str, str], Tensor]
        ret_phis = {}  # Dict[Tuple[str, str], Tensor]
        for key in obj_pair_list:
            if (not all_phis) and obj_pair_list.count(key) > 1:
                continue
            # vmap doesn't support in-place operations
            # ret_contact_forces[key] = torch.zeros(batch_dims + (1, 3))
            # ret_contact_normals[key] = torch.zeros(batch_dims + (1, 3))
            indices = [i for i, x in enumerate(obj_pair_list) if x == key]
            for index_idx, index in enumerate(indices):
                dict_key = key if index_idx == 1 else (key, index_idx)
                ret_phis[dict_key] = m_phi[..., index].reshape(batch_dims + (1, 1))

        ## Populate Contact Normals
        for key in obj_pair_list:
            if (not all_phis) and obj_pair_list.count(key) > 1:
                continue
            indices = np.array([i for i, x in enumerate(obj_pair_list) if x == key])
            if len(indices) == 0:
                continue
            for index_idx, index in enumerate(indices):
                dict_key = key if index_idx == 1 else (key, index_idx)
                fric_index = len(obj_pair_list) + 2 * index
                mr_wf_i = mr_fw_list[index].transpose(-1, -2)
                ret_contact_normals[dict_key] = mr_wf_i[..., :, 2].unsqueeze(-2)

        if no_sim:
            return (
                None,
                None,
                ret_contact_normals,
                ret_phis,
            )

        ## Solve Impulses
        reorder_mat = sappy_reorder_mat(n_contacts)
        reorder_mat = reorder_mat.reshape(
            (1,) * (m_delassus.dim() - 2) + reorder_mat.shape
        ).expand(m_delassus.shape)
        impulse_full = pbmm(
            reorder_mat,
            self._solver(
                pbmm(
                    reorder_mat.transpose(-1, -2), pbmm(mq_delassus, reorder_mat)
                ),  # Quadratic Term
                pbmm(reorder_mat.transpose(-1, -2), q_full).squeeze(-1),  # Linear Term
            ).unsqueeze(-1),
        )

        impulse = torch.nan_to_num(impulse_full * contact_filter)
        # vmap doesn't support dynamic shape
        # impulse = torch.zeros_like(impulse_full)
        # impulse[contact_filter] += impulse_full[contact_filter]

        # pylint doesn't know about torch
        # pylint: disable-next=not-callable
        step_v_add = torch.linalg.solve(
            m_mass, pbmm(m_jac.transpose(-1, -2), impulse)
        ).squeeze(-1)

        ### Populate contact forces
        for key in obj_pair_list:
            if (not all_phis) and obj_pair_list.count(key) > 1:
                continue
            indices = np.array([i for i, x in enumerate(obj_pair_list) if x == key])
            if len(indices) == 0:
                continue
            for index_idx, index in enumerate(indices):
                dict_key = key if index_idx == 1 else (key, index_idx)
                fric_index = len(obj_pair_list) + 2 * index
                mr_wf_i = mr_fw_list[index].transpose(-1, -2)
                # Force in contact_frame
                ret_contact_force_norm = impulse[..., index, 0] * torch.reciprocal(
                    dt
                ).squeeze(-1)
                ret_contact_force_fric = (
                    mu_list[index]
                    * impulse[..., fric_index : fric_index + 2, 0]
                    * torch.reciprocal(dt)
                )
                ret_contact_force = torch.cat(
                    [ret_contact_force_fric, ret_contact_force_norm.unsqueeze(-1)],
                    dim=-1,
                ).unsqueeze(-2)
                # Rotate into world frame
                ret_contact_forces[dict_key] = pbmm(
                    mr_wf_i, ret_contact_force.transpose(-1, -2)
                ).transpose(-1, -2)
        ###
        return (
            step_v_minus + step_v_add,
            ret_contact_forces,
            ret_contact_normals,
            ret_phis,
        )

    def forward_dynamics(
        self,
        step_q: Tensor,
        step_v: Tensor,
        step_u: Tensor,
        step_dt: Optional[float | Tensor] = None,
        solver_torch_vmap: bool = False,
        solver=None,
    ) -> Tensor:
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
            step_q: (\*, space.n_q) current configuration batch.
            step_v: (\*, space.n_v) current velocity batch.
            step_u: (\*, ?) current control batch.
            step_dt: (\*, 1) delta t

        Returns:
            (\*, space.n_v) delta velocity batch.
        """
        # pylint: disable=too-many-locals\
        if solver is None:
            solver = self._solver_learn

        # Input Validation
        batch_dims = step_q.size()[:-1]
        assert step_q.size() == batch_dims + (self.space.n_q,)
        assert step_v.size() == batch_dims + (self.space.n_v,)
        assert step_u.size() == batch_dims + (self._controlled_space.n_v,)
        dt = self._hyperparameters.default_dt * torch.ones(batch_dims + (1,))
        if step_dt is not None:
            step_dt_expanded = step_dt.expand(batch_dims + (1,))
            dt[step_dt_expanded < self._hyperparameters.dt_thresh] = step_dt_expanded[
                step_dt_expanded < self._hyperparameters.dt_thresh
            ]

        assert dt.size() == batch_dims + (1,)
        phi_eps = 1e-2
        eps = 1e-8
        (
            m_delassus,
            m_mass,
            m_jac,
            m_phi,
            non_contact_acceleration,
            obj_pair_list,
            mr_fw_list,
            mu_list,
        ) = self._multibody_terms(step_q, step_v, step_u)
        n_contacts = m_phi.shape[-1]
        contact_filter = (broadcast_lorentz(m_phi) <= phi_eps).unsqueeze(-1)

        mq_delassus = m_delassus + eps * torch.eye(3 * n_contacts)

        # pylint: disable=E1103
        double_zero_vector = torch.zeros(m_phi.shape[:-1] + (2 * n_contacts,))
        phi_then_zero = torch.cat((m_phi, double_zero_vector), dim=-1).unsqueeze(-1)

        step_v_minus = step_v + dt * non_contact_acceleration
        q_full = (
            pbmm(m_jac, step_v_minus.unsqueeze(-1))
            + torch.reciprocal(dt).unsqueeze(-1) * phi_then_zero
        )

        ### Construct contact forces / normals
        ret_contact_forces = {}  # Dict[Tuple[str, str], Tensor]
        ret_contact_normals = {}  # Dict[Tuple[str, str], Tensor]
        ret_phis = {}  # Dict[Tuple[str, str], Tensor]
        for key in obj_pair_list:
            if obj_pair_list.count(key) == 1:
                ret_contact_forces[key] = torch.zeros(batch_dims + (1, 3))
                ret_contact_normals[key] = torch.zeros(batch_dims + (1, 3))
                index = np.array([i for i, x in enumerate(obj_pair_list) if x == key])[
                    0
                ]
                ret_phis[key] = m_phi[..., index].reshape(batch_dims + (1, 1))
        # TODO: Can safely return 0s if above threshold
        # if dt > self._hyperparameters.dt_thresh:
        #    return (
        #        torch.zeros_like(step_v),
        #        ret_contact_forces,
        #        ret_contact_normals,
        #        ret_phis,
        #    )

        ## Solve Impulses
        reorder_mat = sappy_reorder_mat(n_contacts)
        reorder_mat = reorder_mat.reshape(
            (1,) * (m_delassus.dim() - 2) + reorder_mat.shape
        ).expand(m_delassus.shape)
        impulse_full = pbmm(
            reorder_mat,
            solver(
                pbmm(
                    reorder_mat.transpose(-1, -2), pbmm(mq_delassus, reorder_mat)
                ),  # Quadratic Term
                pbmm(reorder_mat.transpose(-1, -2), q_full).squeeze(-1),  # Linear Term
                torch_vmap=solver_torch_vmap,
            ).unsqueeze(-1),
        )

        impulse = torch.zeros_like(impulse_full)
        # TODO: Hack remove NaNs
        impulse[contact_filter] += torch.nan_to_num(impulse_full[contact_filter])

        # pylint doesn't know about torch
        # pylint: disable-next=not-callable
        step_v_add = torch.linalg.solve(
            m_mass, pbmm(m_jac.transpose(-1, -2), impulse)
        ).squeeze(-1)

        ### Populate contact forces
        for key, ret_contact_force in ret_contact_forces.items():
            indices = np.array([i for i, x in enumerate(obj_pair_list) if x == key])
            if len(indices) == 0:
                continue
            assert len(indices) == 1
            index = indices[0]
            fric_index = len(obj_pair_list) + 2 * index
            mr_wf_i = mr_fw_list[index].transpose(-1, -2)
            ret_contact_normals[key][..., 0, :] = mr_wf_i[..., :, 2]
            # Force in contact_frame
            ret_contact_force[..., 0, 2] = impulse[..., index, 0] * torch.reciprocal(
                dt
            ).squeeze(-1)
            ret_contact_force[..., 0, :2] = (
                mu_list[index]
                * impulse[..., fric_index : fric_index + 2, 0]
                * torch.reciprocal(dt)
            )
            # Rotate into world frame
            ret_contact_forces[key] = pbmm(
                mr_wf_i, ret_contact_force.transpose(-1, -2)
            ).transpose(-1, -2)
        ###

        # Zero out bad dts
        try:
            mask = (step_dt_expanded < self._hyperparameters.dt_thresh).float()
            step_v_minus *= mask
            step_v_add *= mask
            for key in ret_contact_forces.keys():
                ret_contact_forces[key] *= mask.unsqueeze(-1)
                ret_contact_normals[key] *= mask.unsqueeze(-1)
                # Keep Phis
        except RuntimeError:
            breakpoint()

        try:
            assert torch.all(~torch.isnan(step_v_minus)), f"NaN in plant_x_batch"
            assert torch.all(~torch.isnan(step_v_add)), f"NaN in plant_x_batch"
        except AssertionError:
            breakpoint()
        return (
            step_v_minus + step_v_add,
            ret_contact_forces,
            ret_contact_normals,
            ret_phis,
        )

    def diff_simulate(
        self,
        ctrl_desired: Tensor,
        timestamps: Optional[Tensor],
        ctrl_actual: Optional[Tensor] = None,
        learned_start_state: Optional[Tensor] = None,
        solver=None,
    ) -> tuple[
        Tensor,
        Tensor,
        dict[tuple[str, str], Tensor],
        dict[tuple[str, str], Tensor],
        dict[tuple[str, str], Tensor],
    ]:
        """
        From the current estimated model state, simulate a batch of robots on the target trajectories.

        Args:
            ctrl_desired: (batch, traj_len, robot.n_x)
            timestamps: (traj_len,)
            ctrl_actual: (batch, traj_len, robot.n_x) (optional: if provided overwrite robot state)
            learned_start_state: (self._learned_trajectory.space.n_x,), use current state if not provided
        Returns:
            - state tensor of size (batch, traj_len, plant.n_x)
            - robot u  (batch, traj_len, robot.n_v)
            - ret_contact_forces Dict[<collision>, size(batch, traj_len, 3)]
            - ret_contact_normals Dict[<collision>, size(batch, traj_len, 3)]
            - ret_contact_phis Dict[<collision>, size(batch, traj_len, 1)]
        """
        # pylint: disable=too-many-locals, too-many-branches, too-many-statements

        # Input Validation
        assert len(ctrl_desired.size()) >= 2
        batch_dims = ctrl_desired.size()[:-2]
        traj_len = ctrl_desired.size()[-2]
        assert ctrl_desired.size() == batch_dims + (
            traj_len,
            self._controlled_space.n_x,
        ), str(ctrl_desired.size())
        timestamps = (
            torch.arange(
                traj_len * self._hyperparameters.default_dt,
                step=self._hyperparameters.default_dt,
            )
            if timestamps is None
            else timestamps
        )
        assert timestamps.size() == (traj_len,)

        assert (ctrl_actual is None) or (
            ctrl_actual.size()
            == batch_dims
            + (
                traj_len,
                self._controlled_space.n_x,
            )
        )

        if learned_start_state is not None:
            assert learned_start_state.size() == (self._learned_trajectory.space.n_x,)
        else:
            learned_start_state = self._learned_trajectory.current_state()

        # print(f"Running DiffSim of Length {traj_len}...")

        # Populate Initial Controlled State
        data_state = TensorDict({}, batch_size=batch_dims + (traj_len,))
        ctrl_desired_splits = self._controlled_space.x_split(ctrl_desired)
        ctrl_actual_splits = (
            None
            if (ctrl_actual is None)
            else self._controlled_space.x_split(ctrl_actual)
        )
        for robot_model_idx, (robot_model_name, robot_model_space) in enumerate(
            zip(self._controlled_model_names, self._controlled_space.spaces)
        ):
            data_state[robot_model_name + "_state"] = (
                robot_model_space.zero_state().repeat(batch_dims + (traj_len, 1))
            )
            data_state[robot_model_name + "_state"][..., 0, :] = (
                ctrl_desired_splits[robot_model_idx][..., 0, :]
                if ctrl_actual_splits is None
                else ctrl_actual_splits[robot_model_idx][..., 0, :]
            )

        # Populate Initial Learned State
        for traj_model_idx, (traj_model_name, traj_model_x) in enumerate(
            zip(
                self._learned_model_names,
                self._learned_trajectory.space.x_split(learned_start_state),
            )
        ):
            assert len(traj_model_x.size()) == 1
            data_state[traj_model_name + "_state"] = (
                self._learned_trajectory.space.spaces[traj_model_idx]
                .zero_state()
                .repeat(batch_dims + (traj_len, 1))
            )
            data_state[traj_model_name + "_state"][..., 0, :] = traj_model_x

        ## Simulation Loop
        ret_contact_forces = {}
        ret_contact_normals = {}
        ret_contact_phis = {}
        ret_u = torch.zeros(batch_dims + (traj_len, self._controlled_space.n_v))
        for sim_idx in range(1, traj_len):
            # print(f"Sim Step {sim_idx} / {traj_len}...")
            sim_dt = timestamps[sim_idx] - timestamps[sim_idx - 1]
            # print(f"Sim DT: {sim_dt}")
            # Calculate u from robot PD
            step_u = []
            for robot_model_idx, (robot_model_name, robot_model_space) in enumerate(
                zip(self._controlled_model_names, self._controlled_space.spaces)
            ):
                robot_model_step_state = data_state[robot_model_name + "_state"][
                    ..., sim_idx - 1, :
                ].detach()
                assert robot_model_step_state.size() == batch_dims + (
                    robot_model_space.n_x,
                )
                robot_model_target_state = ctrl_desired_splits[robot_model_idx][
                    ..., sim_idx - 1, :
                ]
                assert robot_model_target_state.size() == batch_dims + (
                    robot_model_space.n_x,
                )
                step_u.append(
                    self._hyperparameters.ctrl_kp
                    * (
                        robot_model_space.q(robot_model_target_state)
                        - robot_model_space.q(robot_model_step_state)
                    )
                    + self._hyperparameters.ctrl_kd
                    * (
                        robot_model_space.v(robot_model_target_state)
                        - robot_model_space.v(robot_model_step_state)
                    )
                )
                assert step_u[-1].size() == batch_dims + (robot_model_space.n_v,)
            # Run Forward Dynamics
            step_q, step_v = self.space.q_v(
                self._multibody_terms.construct_state_tensor(
                    data_state[..., sim_idx - 1]
                )
            )
            try:
                assert torch.all(~torch.isnan(step_q)), f"NaN in step_q"
                assert torch.all(~torch.isnan(step_v)), f"NaN in step_v"
            except AssertionError:
                breakpoint()

            ### TODO: HACK clip step_v magnitude
            step_v = torch.clamp(step_v, min=-1e3, max=1e3)

            # print(f"With robot_state: {data_state[..., sim_idx - 1]["robot_state"].detach().cpu().numpy()}")
            # print(f"With cube_state: {data_state[..., sim_idx - 1]["cube_state"].detach().cpu().numpy()}")
            step_vplus, step_contact_forces, step_contact_normals, step_contact_phis = (
                self.forward_dynamics(
                    step_q, step_v, torch.cat(step_u, dim=-1), sim_dt, solver=solver
                )
            )
            data_state_vec = self.space.x(
                self.space.euler_step(step_q, step_vplus, sim_dt),
                step_vplus,
            )
            try:
                assert torch.all(~torch.isnan(data_state_vec)), f"NaN in data_state_vec"
            except AssertionError:
                breakpoint()
            data_state[..., sim_idx] = (
                self._multibody_terms.model_states_from_state_tensor(data_state_vec)
            )
            # Overwrite actual robot state
            if ctrl_actual_splits is not None:
                for robot_model_idx, robot_model_name in enumerate(
                    self._controlled_model_names
                ):
                    data_state[robot_model_name + "_state"][..., sim_idx, :] = (
                        ctrl_actual_splits[robot_model_idx][..., sim_idx, :]
                    )

            # Record contact forces, normals, and phis
            for key, value in step_contact_forces.items():
                if key not in ret_contact_forces:
                    ret_contact_forces[key] = torch.zeros(
                        batch_dims + (traj_len - 1, 3)
                    )
                ret_contact_forces[key][..., sim_idx - 1, :] += value[..., 0, :]
            for key, value in step_contact_normals.items():
                if key not in ret_contact_normals:
                    ret_contact_normals[key] = torch.zeros(
                        batch_dims + (traj_len - 1, 3)
                    )
                ret_contact_normals[key][..., sim_idx - 1, :] = value[..., 0, :]
            for key, value in step_contact_phis.items():
                if key not in ret_contact_phis:
                    ret_contact_phis[key] = torch.zeros(batch_dims + (traj_len, 1))
                ret_contact_phis[key][..., sim_idx - 1, :] = value[..., 0, :]
            # Record robot control
            ret_u[..., sim_idx, :] = torch.cat(step_u, dim=-1)

        # Record final phi
        step_q, step_v = self.space.q_v(
            self._multibody_terms.construct_state_tensor(data_state[..., traj_len - 1])
        )
        (
            _,
            _,
            _,
            final_phi,
            _,
            obj_pair_list,
            _,
            _,
        ) = self._multibody_terms(
            step_q, step_v, torch.zeros(batch_dims + (self._controlled_space.n_v,))
        )
        for key in obj_pair_list:
            indices = np.array([i for i, x in enumerate(obj_pair_list) if x == key])
            if len(indices) != 1:
                continue
            ret_contact_phis[key][..., traj_len - 1, 0] = final_phi[..., indices[0]]

        try:
            test_out = self._multibody_terms.construct_state_tensor(data_state).detach()
            assert torch.all(~torch.isnan(test_out)), f"NaN in plant_x_batch"
        except AssertionError:
            breakpoint()

        return (
            self._multibody_terms.construct_state_tensor(data_state),
            ret_u,
            ret_contact_forces,
            ret_contact_normals,
            ret_contact_phis,
        )

    @override
    def forward(
        self,
        ctrl_desired: Tensor,
        timestamps: Optional[Tensor] = None,
        ctrl_actual: Optional[Tensor] = None,
        nimp_override: bool = False,
    ) -> tuple[
        Tensor,
        Tensor,
        Optional[dict[tuple[str, str], Tensor]],
        Optional[dict[tuple[str, str], Tensor]],
        Optional[dict[tuple[str, str], Tensor]],
    ]:
        """Forward Function for the Learnable System

        If NIMP (or nimp_override), use diffsim to get system state. Otherwise concat
        from learnable trajectory.

        Args:
            ctrl_desired: Input desired robot trajectory of size (batch, traj_len, self._controlled_space.n_x)
            timestamps: (traj_len,), if not provided use hyperparameters.default_dt
            ctrl_actual: Actual robot trajectory of size (batch, traj_len, self._controlled_space.n_x)
            nimp_override: If true, run diffsim from current object pose.

        Returns:
            system_state (batch, traj_len, model_space.n_x)
            system_control (batch, traj_len-1, controlled_space.n_v)
            (nimp only) forces Dict[obj_pair_str, (batch, traj_len-1, 3)]
            (nimp only) normals Dict[obj_pair_str, (batch, traj_len-1, 3)]
            (nimp only) phi Dict[obj_pair_str, (batch, traj_len, 1)]

        """

        # pylint: disable=too-many-locals

        # Input Validation
        assert len(ctrl_desired.size()) >= 2
        batch_dims = ctrl_desired.size()[:-2]
        traj_len = ctrl_desired.size()[-2]
        assert ctrl_desired.size() == batch_dims + (
            traj_len,
            self._controlled_space.n_x,
        ), str(ctrl_desired.size())
        timestamps = (
            torch.arange(
                traj_len * self._hyperparameters.default_dt,
                step=self._hyperparameters.default_dt,
            )
            if timestamps is None
            else timestamps
        )
        assert timestamps.size() == (traj_len,)

        # Nan checks
        assert ~torch.any(torch.isnan(ctrl_desired))
        if ctrl_actual is not None:
            assert ~torch.any(torch.isnan(ctrl_actual))

        # Naive Implicit Loss
        if self._hyperparameters.loss_fn == LossFunction.NIMP or nimp_override:
            # Run Differential Simulation
            return self.diff_simulate(ctrl_desired, timestamps, ctrl_actual)

        # Violaiton Implicit Loss
        assert (
            len(self._learned_trajectory) == traj_len
        ), f"Traj Len doesn't match: {len(self._learned_trajectory)} vs. {traj_len}"

        data_state = TensorDict({}, batch_size=batch_dims + (traj_len,))
        # Populate Learned Trajectory
        for traj_model_name, traj_model_x in zip(
            self._learned_model_names,
            self._learned_trajectory.space.x_split(
                self._learned_trajectory.get_current_traj(pose_only=False)
            ),
        ):
            assert len(traj_model_x.size()) == 2
            data_state[traj_model_name + "_state"] = traj_model_x.repeat(
                batch_dims + (1, 1)
            )

        # Populate robot state
        ctrl_desired_splits = self._controlled_space.x_split(ctrl_desired)
        ctrl_actual_splits = (
            None if ctrl_actual is None else self._controlled_space.x_split(ctrl_actual)
        )
        control_u = []
        for robot_model_idx, (robot_model_name, robot_model_space) in enumerate(
            zip(self._controlled_model_names, self._controlled_space.spaces)
        ):
            data_state[robot_model_name + "_state"] = (
                ctrl_desired_splits[robot_model_idx]
                if ctrl_actual_splits is None
                else ctrl_actual_splits[robot_model_idx]
            )

            # Calculate Control
            if ctrl_actual_splits is not None:
                control_u.append(
                    self._hyperparameters.ctrl_kp
                    * (
                        robot_model_space.q(ctrl_desired_splits[robot_model_idx])
                        - robot_model_space.q(ctrl_actual_splits[robot_model_idx])
                    )
                    + self._hyperparameters.ctrl_kd
                    * (
                        robot_model_space.v(ctrl_desired_splits[robot_model_idx])
                        - robot_model_space.v(ctrl_actual_splits[robot_model_idx])
                    )
                )
            else:
                control_u.append(
                    torch.zeros(batch_dims + (traj_len, robot_model_space.n_v))
                )
            assert control_u[-1].size() == batch_dims + (
                traj_len,
                robot_model_space.n_v,
            )

        return (
            self._multibody_terms.construct_state_tensor(data_state),
            torch.cat(control_u, dim=-1),
            None,
            None,
            None,
        )

    def _loss_nimp(
        self,
        meas_contact_forces: Optional[dict[tuple[str, str], Tensor]],
        meas_contact_normals: dict[tuple[str, str], Tensor],
        est_contact_forces: dict[tuple[str, str], Tensor],
        est_contact_normals: dict[tuple[str, str], Tensor],
        est_contact_phis: dict[tuple[str, str], Tensor],
    ) -> dict[str, Tensor]:
        """
        NIMP Loss
        contact_normal = 0 means that no contact is detected.
        if len(meas_contact_forces/normals) > traj_len-1 (could be traj_len),
            only use the first traj_len-1

        Args:
            meas_contact_forces: Dict[<collision>, size(batch, traj_len-1, 3)] in world frame
            meas_contact_normals: Dict[<collision>, size(batch, traj_len-1, 3)] in world frame
            est_contact_forces: Dict[<collision>, size(batch, traj_len-1, 3)] in world frame
            est_contact_normals: Dict[<collision>, size(batch, traj_len-1, 3)] in world frame
            est_contact_phis: Dict[<collision>, size(batch, traj_len, 1)] SDF
        Returns:
            Dictionary of loss terms, each identified by a string.
            Each term is pre-scaled and comes in size(batch, traj_len-1)
            Loss Terms Are:
             * Contact Boolean Measurement (loss_meas_bool)
             * Contact Force Measurement (loss_meas_force)
             * Contact Normal Measurement (loss_meas_normal)

        """

        # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals

        # Add reverse keys to make access easier
        # Forces and Normals Flip Signs
        for in_dict in (
            est_contact_normals,
            est_contact_forces,
        ):
            for key in list(in_dict):
                revkey = (key[1], key[0])
                assert revkey not in in_dict, f"Key and Reverse Key {key} in dictionary"
                in_dict[revkey] = -in_dict[key]
        # SDF doesn't flip signs
        for key in list(est_contact_phis):
            revkey = (key[1], key[0])
            assert (
                revkey not in est_contact_phis
            ), f"Key and Reverse Key {key} in dictionary"
            est_contact_phis[revkey] = est_contact_phis[key]

        supervised_keys = list(set(meas_contact_normals) & set(est_contact_phis))

        # Input Validation
        for key in supervised_keys:
            assert len(est_contact_phis[key].size()) >= 2, est_contact_phis[key].size()
            batch_dims = est_contact_phis[key].size()[:-2]
            traj_len = est_contact_phis[key].size()[-2]
            assert est_contact_phis[key].size()[-1] == 1, est_contact_phis[key].size()
            assert est_contact_normals[key].size() == batch_dims + (
                traj_len - 1,
                3,
            ), est_contact_normals[key].size()
            assert est_contact_forces[key].size() == batch_dims + (
                traj_len - 1,
                3,
            ), est_contact_forces[key].size()
            assert meas_contact_normals[key].size() == batch_dims + (
                traj_len - 1,
                3,
            ), meas_contact_normals[key].size()
            if meas_contact_forces is not None:
                assert meas_contact_forces[key].size() == batch_dims + (
                    traj_len - 1,
                    3,
                ), meas_contact_forces[key].size()

        ret_loss = {
            "loss_meas_bool": torch.zeros(batch_dims + (traj_len - 1,)),
            "loss_meas_force": torch.zeros(batch_dims + (traj_len - 1,)),
            "loss_meas_normal": torch.zeros(batch_dims + (traj_len - 1,)),
            "loss_pen": torch.zeros(batch_dims + (traj_len - 1,)),
        }

        # Supervise each key
        for key in supervised_keys:
            # Input Validation
            if meas_contact_forces is not None:
                assert (
                    key in meas_contact_forces
                ), f"Key {key} not in meas_contact_forces"
            assert key in meas_contact_normals, f"Key {key} not in meas_contact_normals"
            assert key in est_contact_forces, f"Key {key} not in est_contact_forces"
            assert key in est_contact_normals, f"Key {key} not in est_contact_normals"
            assert key in est_contact_phis, f"Key {key} not in est_contact_phis"
            contact_bool = torch.ones(batch_dims + (traj_len - 1,))
            contact_bool[
                torch.isclose(
                    # pylint doesn't know about torch
                    # pylint: disable-next=not-callable
                    torch.linalg.vector_norm(meas_contact_normals[key], dim=-1),
                    torch.zeros(batch_dims + (traj_len - 1,)),
                )
            ] = 0.0

            # Contact Bool Loss
            phi_alpha = (
                np.log((1.0 / self._hyperparameters.w_phi_ci) - 1.0)
                / self._hyperparameters.w_phi_nominal
            )
            loss_meas_bool = (contact_bool - 1.0) * phi_alpha * est_contact_phis[key][
                ..., 1:, 0
            ] + torch.log(
                1.0 + torch.exp(phi_alpha * est_contact_phis[key][..., 1:, 0])
            )
            assert loss_meas_bool.size() == batch_dims + (
                traj_len - 1,
            ), loss_meas_bool.size()
            ret_loss["loss_meas_bool"] += loss_meas_bool

            # Normal Loss
            loss_meas_normal = (
                0.5
                * contact_bool
                * (1.0 / self._hyperparameters.w_normal_var)
                * (
                    1.0
                    - pbmm(
                        est_contact_normals[key].unsqueeze(-2),
                        meas_contact_normals[key].unsqueeze(-1),
                    )
                    .squeeze(-2)
                    .squeeze(-1)
                )
            )
            assert loss_meas_normal.size() == batch_dims + (
                traj_len - 1,
            ), loss_meas_normal.size()
            ret_loss["loss_meas_normal"] += loss_meas_normal

            # Force Loss
            if meas_contact_forces is not None:
                loss_meas_force = (
                    0.5
                    * (
                        torch.ones_like(contact_bool)
                        if self._hyperparameters.supervise_non_contact_force
                        else contact_bool
                    )
                    * (1.0 / self._hyperparameters.w_force_var)
                    * pbmm(
                        (est_contact_forces[key] - meas_contact_forces[key]).unsqueeze(
                            -2
                        ),
                        (est_contact_forces[key] - meas_contact_forces[key]).unsqueeze(
                            -1
                        ),
                    )
                    .squeeze(-2)
                    .squeeze(-1)
                )
                assert loss_meas_force.size() == batch_dims + (
                    traj_len - 1,
                ), loss_meas_force.size()
                ret_loss["loss_meas_force"] += loss_meas_force

            # Penetration Loss (start only for NIMP, assume sim avoids penetration)
            ret_loss["loss_pen"] += self._hyperparameters.w_pen * torch.maximum(
                -est_contact_phis[key][..., 0, 0],
                torch.zeros_like(est_contact_phis[key][..., 0, 0]),
            )

        return ret_loss

    def state_map_for_learnable_bodies(
        self,
    ) -> Tensor:
        """Returns a boolean tensor indicating which states correspond to
        learnable bodies.
        """
        # TODO: Make agnostic to number of robot models
        assert len(self._controlled_model_names) == 1
        return torch.tensor(
            [
                not s.startswith(self._controlled_model_names[0])
                for s in self.plant.GetStateNames()
            ]
        ).detach()

    def _loss_vimp(
        self,
        meas_contact_forces: Optional[dict[tuple[str, str], Tensor]],
        meas_contact_normals: dict[tuple[str, str], Tensor],
        timestamps: Tensor,
        plant_x: Tensor,
        plant_u: Tensor,
    ) -> dict[str, Tensor]:
        """
        VIMP Loss
        contact_normal = 0 means that no contact is detected.
        if len(meas_contact_forces/normals) > traj_len-1 (could be traj_len),
            only use the first traj_len-1

        Args:
            meas_contact_forces: Dict[<collision>, size(batch, traj_len-1, 3)] in world frame
            meas_contact_normals: Dict[<collision>, size(batch, traj_len-1, 3)] in world frame
            timestamps: size(traj_len,)
            plant_x: size(batch, traj_len, space.n_x)
            plant_u: size(batch, traj_len, controlled_space.n_v)
        Returns:
            Dictionary of loss terms, each identified by a string.
            Each term is pre-scaled and comes in size(batch, traj_len-1)
            lambda-dependent Loss Terms Are:
             * Prediction: Velocity (loss_v_pred)
             * Complementarity (loss_comp)
             * Max Power Dissipation (loss_diss)
             * Inelasticity (loss_elas)
             * Contact Force Measurement (loss_meas_force)
            lambda-independent Loss Terms Are:
             * Contact Boolean Measurement (loss_meas_bool)
             * Contact Normal Measurement (loss_meas_normal)
             * Prediction: Position (loss_q_pred)
             * Penetration (loss_pen)
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments,
        # pylint: disable=too-many-statements, too-many-locals

        # TODO: Make Hyperparameter
        eps = 1e-5

        # Input Validation
        batch_dims = plant_x.size()[:-2]
        traj_len = plant_x.size()[-2]
        assert timestamps.size() == (traj_len,)
        for key in list(meas_contact_normals):
            assert meas_contact_normals[key].size() == batch_dims + (
                traj_len - 1,
                3,
            ), meas_contact_normals[key].size()
            if meas_contact_forces is not None:
                assert meas_contact_forces[key].size() == batch_dims + (
                    traj_len - 1,
                    3,
                ), meas_contact_forces[key].size()
        assert plant_x.size() == batch_dims + (traj_len, self.space.n_x), plant_x.size()
        assert plant_u.size() == batch_dims + (
            traj_len,
            self._controlled_space.n_v,
        ), plant_u.size()

        # Nan Checks
        assert ~torch.any(torch.isnan(plant_x))
        assert ~torch.any(torch.isnan(plant_u))

        ret_loss = {
            "loss_meas_bool": torch.zeros(batch_dims + (traj_len - 1,)),
            "loss_meas_force": torch.zeros(batch_dims + (traj_len - 1,)),
            "loss_meas_normal": torch.zeros(batch_dims + (traj_len - 1,)),
            "loss_pen": torch.zeros(batch_dims + (traj_len - 1,)),
            "loss_v_pred": torch.zeros(batch_dims + (traj_len - 1,)),
            "loss_q_pred": torch.zeros(batch_dims + (traj_len - 1,)),
            "loss_comp": torch.zeros(batch_dims + (traj_len - 1,)),
            "loss_diss": torch.zeros(batch_dims + (traj_len - 1,)),
            "loss_elas": torch.zeros(batch_dims + (traj_len - 1,)),
        }

        # Compute DTs, switch trajectory jumps to default_dt
        dts = (timestamps[1:] - timestamps[:-1]).unsqueeze(-1)
        dts[dts > self._hyperparameters.dt_thresh] = self._hyperparameters.default_dt

        # Extract multibody terms, use next state and current control
        plant_xplus = plant_x[..., 1:, :]
        plant_xmin = plant_x[..., :-1, :]
        (
            m_delassus,
            m_mass,
            m_jac,
            m_phi,
            non_contact_acceleration,
            obj_pair_list,
            mr_fw_list,
            mu_list,
        ) = self._multibody_terms(
            self.space.q(plant_xplus), self.space.v(plant_xplus), plant_u[..., :-1, :]
        )
        n_contacts = m_phi.shape[-1]

        ### Prediction: Velocity Term
        # Exclude robot predictions
        # velocity_mask: 1 for learnable velocities, 0 for robot velocities.
        velocity_mask = self.state_map_for_learnable_bodies()[self.space.n_q :]
        plant_dv = (
            self.space.v(plant_xplus)
            - (self.space.v(plant_xmin) + non_contact_acceleration * dts)
        ).unsqueeze(-2)

        qp_v_pred = pbmm(
            m_jac[..., velocity_mask],
            pbmm(
                torch.inverse(m_mass[..., velocity_mask, :][..., velocity_mask]),
                m_jac[..., velocity_mask].transpose(-1, -2),
            ),
        ) + eps * torch.eye(
            3 * n_contacts
        )  # Units: Energy. Must be positive-definite
        q_v_pred = -pbmm(
            m_jac[..., velocity_mask], plant_dv[..., velocity_mask].transpose(-1, -2)
        )
        const_v_pred = 0.5 * pbmm(
            plant_dv[..., velocity_mask],
            pbmm(
                m_mass[..., velocity_mask, :][..., velocity_mask],
                plant_dv[..., velocity_mask].transpose(-1, -2),
            ),
        )

        ### Complementarity
        # 0-pad in friction terms
        phi_then_zero = torch.cat(
            (m_phi, torch.zeros(m_phi.shape[:-1] + (2 * n_contacts,))), dim=-1
        )
        q_comp = (
            torch.reciprocal(dts)
            * torch.maximum(phi_then_zero, torch.zeros_like(phi_then_zero))
        ).unsqueeze(-1)

        ### Max Power Dissipation (loss_diss)
        sliding_velocities = pbmm(
            m_jac[..., n_contacts:, :], self.space.v(plant_xplus).unsqueeze(-1)
        )

        # Need non-0 norm for Hessian calculation, hence add eps to norm()
        sliding_speeds = (
            sliding_velocities.reshape(m_phi.shape[:-1] + (n_contacts, 2)) + eps
        ).norm(dim=-1, keepdim=True)

        q_diss = torch.cat((sliding_speeds, sliding_velocities), dim=-2)

        ### Inelasticity (loss_elas)
        normal_velocities = pbmm(
            m_jac[..., :n_contacts, :], self.space.v(plant_xplus).unsqueeze(-1)
        )
        normal_velocities = torch.maximum(
            normal_velocities, torch.zeros_like(normal_velocities)
        )
        # 0-pad in friction terms
        q_elas = torch.cat(
            (
                normal_velocities,
                torch.zeros(m_phi.shape[:-1] + (2 * n_contacts,)).unsqueeze(-1),
            ),
            dim=-2,
        )

        ### Contact Force Measurement (loss_meas_force)
        q_meas_force = torch.zeros_like(q_v_pred)
        qp_meas_force = torch.zeros_like(qp_v_pred)
        const_meas_force = torch.zeros_like(const_v_pred)

        if meas_contact_forces is not None:
            for key in meas_contact_forces.keys():
                indices = np.array([i for i, x in enumerate(obj_pair_list) if x == key])
                if len(indices) == 0:
                    continue
                mu_i = mu_list[indices[0]]
                # qp_meas_force = diag(mu)RS^TSR^Tdiag(mu)^T; diag(mu) = 1 if normal, mu otherwise
                # R is block diagonal rotation matrices, S is summation matrix
                diag_f_mu = torch.zeros(
                    q_meas_force.shape[:-1] + ((len(indices) * 3),)
                )  # (batch x (n_c_tot*3) x (n_c_obj*3))
                r_fw_mat = torch.zeros(
                    q_meas_force.shape[:-2] + ((len(indices) * 3), (len(indices) * 3))
                )  # (batch x (n_c_obj*3) x (n_c_obj*3))
                sum_w_mat = torch.zeros(
                    q_meas_force.shape[:-2] + (3, (len(indices) * 3))
                )  # (batch x 3 x (n_c_obj*3))
                for contact, idx in enumerate(indices):
                    # Map Normal Force
                    diag_f_mu[..., idx, contact * 3 + 2] = 1.0
                    # Map Tangent Forces
                    diag_f_mu[..., len(obj_pair_list) + 2 * idx, contact * 3] = mu_i
                    diag_f_mu[
                        ..., len(obj_pair_list) + 2 * idx + 1, contact * 3 + 1
                    ] = mu_i

                    # Create Block diagonal matrix (note torch.block_diag isn't vectorized)
                    r_fw_mat[
                        ...,
                        contact * 3 : (contact + 1) * 3,
                        contact * 3 : (contact + 1) * 3,
                    ] = mr_fw_list[idx]

                    # Summation
                    sum_w_mat[..., 0, contact * 3] = 1.0
                    sum_w_mat[..., 1, contact * 3 + 1] = 1.0
                    sum_w_mat[..., 2, contact * 3 + 2] = 1.0
                q_meas_force_part = pbmm(
                    sum_w_mat,
                    pbmm(r_fw_mat.transpose(-1, -2), diag_f_mu.transpose(-1, -2)),
                )  # (batch, 3, (n_c_tot*3))
                assert q_meas_force_part.size() == batch_dims + (
                    traj_len - 1,
                    3,
                    n_contacts * 3,
                )
                qp_meas_force += pbmm(
                    q_meas_force_part.transpose(-1, -2), q_meas_force_part
                )

                # Linear Term is lambda_mSR^Tdiag(mu)^T
                impulse_measured = (meas_contact_forces[key] * dts).unsqueeze(
                    -2
                )  # (batch, 1, 3)
                assert impulse_measured.size() == batch_dims + (traj_len - 1, 1, 3)
                q_meas_force -= pbmm(impulse_measured, q_meas_force_part).transpose(
                    -1, -2
                )  # (batch, n_c_tot*3, 1)

                # Constant term is lambda_m magnitude, multiply by 0.5 here to match constant_pred
                const_meas_force += 0.5 * pbmm(
                    impulse_measured, impulse_measured.transpose(-1, -2)
                )

        qp_final = (
            self._hyperparameters.w_v_pred * qp_v_pred
            + (1.0 / self._hyperparameters.w_force_var) * qp_meas_force
        )

        q_final = (
            self._hyperparameters.w_v_pred * q_v_pred
            + self._hyperparameters.w_comp * q_comp
            + self._hyperparameters.w_diss * q_diss
            + self._hyperparameters.w_elas * q_elas
            + (1.0 / self._hyperparameters.w_force_var) * q_meas_force
        )

        # Envelope theorem guarantees that gradient of loss w.r.t. parameters
        # can ignore the gradient of the impulses w.r.t. the QCQP parameters.
        # Therefore, we can detach ``impulses`` from pytorch's computation graph
        # without causing error in the overall loss gradient.
        # Construct a reordering matrix s.t. lambda_CN = reorder_mat @ f_sappy.
        reorder_mat = sappy_reorder_mat(n_contacts)
        reorder_mat = reorder_mat.reshape(
            (1,) * (m_delassus.dim() - 2) + reorder_mat.shape
        ).expand(m_delassus.shape)
        with torch.no_grad():
            impulses = pbmm(
                reorder_mat,
                self._solver_learn(
                    pbmm(
                        reorder_mat.transpose(-1, -2), pbmm(qp_final, reorder_mat)
                    ),  # Quadratic Term
                    pbmm(reorder_mat.transpose(-1, -2), q_final).squeeze(
                        -1
                    ),  # Linear Term
                ).unsqueeze(-1),
            )

        # Hack: remove elements of ``impulses`` where solver likely failed.
        invalid = torch.any(
            (impulses.abs() > 1e3) | impulses.isnan() | impulses.isinf(),
            dim=-2,
            keepdim=True,
        )
        impulses[invalid.expand(impulses.shape)] = 0.0
        # Zero out negative normals (possible within solver tolerance)
        impulses[..., :n_contacts, 0] = torch.maximum(
            impulses[..., :n_contacts, 0].clone(),
            torch.zeros_like(impulses[..., :n_contacts, 0].detach()),
        ).clone()
        const_v_pred[invalid] *= 0.0
        const_meas_force[invalid] *= 0.0

        ### Loss: Prediction: Velocity (loss_v_pred)
        ret_loss["loss_v_pred"] += self._hyperparameters.w_v_pred * (
            0.5 * pbmm(impulses.transpose(-1, -2), pbmm(qp_v_pred, impulses))
            + pbmm(impulses.transpose(-1, -2), q_v_pred)
            + const_v_pred
        ).squeeze(-1).squeeze(-1)
        try:
            assert ret_loss["loss_v_pred"].size() == batch_dims + (traj_len - 1,)
            assert np.all(
                ret_loss["loss_v_pred"].detach().cpu().numpy() >= -eps
            ), f"Velocity Prediction Loss Negative: {np.min(ret_loss["loss_v_pred"].detach().cpu().numpy())}"
        except AssertionError as error:
            print(f"WARNING: {error}")

        ### Loss: Complementarity (loss_comp)
        ret_loss["loss_comp"] += self._hyperparameters.w_comp * pbmm(
            impulses.transpose(-1, -2), q_comp
        ).squeeze(-1).squeeze(-1)
        assert ret_loss["loss_comp"].size() == batch_dims + (traj_len - 1,)

        ### Loss: Max Power Dissipation (loss_diss)
        ret_loss["loss_diss"] += self._hyperparameters.w_diss * pbmm(
            impulses.transpose(-1, -2), q_diss
        ).squeeze(-1).squeeze(-1)
        assert ret_loss["loss_diss"].size() == batch_dims + (traj_len - 1,)

        ### Loss: Inelasticity (loss_elas)
        ret_loss["loss_elas"] += self._hyperparameters.w_elas * pbmm(
            impulses.transpose(-1, -2), q_elas
        ).squeeze(-1).squeeze(-1)
        assert ret_loss["loss_elas"].size() == batch_dims + (traj_len - 1,)

        ### Loss: Contact Force Measurement (loss_meas_force)
        ret_loss["loss_meas_force"] += (1.0 / self._hyperparameters.w_force_var) * (
            0.5 * pbmm(impulses.transpose(-1, -2), pbmm(qp_meas_force, impulses))
            + pbmm(impulses.transpose(-1, -2), q_meas_force)
            + const_meas_force
        ).squeeze(-1).squeeze(-1)
        assert ret_loss["loss_meas_force"].size() == batch_dims + (traj_len - 1,)
        assert np.all(
            ret_loss["loss_meas_force"].detach().cpu().numpy() >= -eps
        ), "Contact Force Measurement Loss Negative"

        ### Loss: Prediction: Position (loss_q_pred)
        vel_err = (
            self.space.configuration_difference(
                self.space.euler_step(
                    self.space.q(plant_xmin), self.space.v(plant_xplus), dts
                ),
                self.space.q(plant_xplus),
            )
            / dts
        )
        # Exclude robot trajectory
        vel_err[..., ~velocity_mask] = 0.0
        ret_loss["loss_q_pred"] += self._hyperparameters.w_q_pred * pbmm(
            vel_err.unsqueeze(-2), pbmm(m_mass, vel_err.unsqueeze(-1))
        ).squeeze(-1).squeeze(-1)
        assert ret_loss["loss_q_pred"].size() == batch_dims + (traj_len - 1,)

        ### Loss: Penetration (loss_pen)
        ret_loss["loss_pen"] += self._hyperparameters.w_pen * torch.maximum(
            -m_phi,
            torch.zeros_like(m_phi),
        ).sum(dim=-1)
        assert ret_loss["loss_pen"].size() == batch_dims + (traj_len - 1,)

        ### Loss: Contact Normal Measurement (loss_meas_normal)
        phi_alpha = (
            np.log((1.0 / self._hyperparameters.w_phi_ci) - 1.0)
            / self._hyperparameters.w_phi_nominal
        )
        for key in meas_contact_normals.keys():
            revkey = (key[1], key[0])
            indices = np.array(
                [i for i, x in enumerate(obj_pair_list) if x == key or x == revkey]
            )
            if len(indices) == 0:
                continue
            for idx in indices:
                objkey = obj_pair_list[idx]
                norm_mult = -1.0 if objkey == revkey else 1.0
                normals_guess = norm_mult * mr_fw_list[idx].transpose(-1, -2)[..., 2]
                assert normals_guess.size() == batch_dims + (traj_len - 1, 3)
                assert meas_contact_normals[key].size() == batch_dims + (
                    traj_len - 1,
                    3,
                )
                contact_bool = torch.ones(batch_dims + (traj_len - 1,))
                contact_bool[
                    torch.isclose(
                        # pylint doesn't know about torch
                        # pylint: disable-next=not-callable
                        torch.linalg.vector_norm(meas_contact_normals[key], dim=-1),
                        torch.zeros(batch_dims + (traj_len - 1,)),
                    )
                ] = 0.0
                # Batch dot product (max 0 for numerical stability)
                cost_normal = torch.maximum(
                    1.0 - (meas_contact_normals[key] * normals_guess).sum(dim=-1),
                    torch.zeros(batch_dims + (traj_len - 1,)),
                )
                assert cost_normal.size() == batch_dims + (traj_len - 1,)
                ret_loss["loss_meas_normal"] += (
                    (0.5 / self._hyperparameters.w_normal_var)
                    * contact_bool
                    * cost_normal
                )
                assert ret_loss["loss_meas_normal"].size() == batch_dims + (
                    traj_len - 1,
                )

                ### Loss: Contact Boolean Measurement (loss_meas_bool)
                # TODO: get rid of log(exp(x)) for large X (replace w/ linear)
                m_phi_close = torch.clamp(
                    m_phi[..., idx], max=self._hyperparameters.w_phi_nominal
                )
                m_phi_far = torch.clamp(m_phi[..., idx] - m_phi_close, min=0.0)
                ret_loss["loss_meas_bool"] += (
                    (contact_bool - 1.0) * phi_alpha * m_phi[..., idx]
                    + torch.log(1.0 + torch.exp(phi_alpha * m_phi_close))
                    + phi_alpha * m_phi_far
                )
                assert ret_loss["loss_meas_bool"].size() == batch_dims + (traj_len - 1,)

        return ret_loss

    def loss_fn(
        self,
        meas_contact_forces,
        meas_contact_normals,
        timestamps: Optional[Tensor],
        *forward_args,
    ) -> dict[str, Tensor]:
        """
        Calculate the loss given some measurement data.
        contact_normal = 0 means that no contact is detected.

        Removes 0th element from measured forces to match trajectory length.

        Args:
            meas_contact_forces: Dict[<collision>, size(batch, traj_len-1, 3)] in world frame
            meas_contact_normals: Dict[<collision>, size(batch, traj_len-1, 3)] in world frame
            timestamps: (traj_len,)
            *forward_args: See return type of forward()
        Returns:
            Dictionary of loss terms, each identified by a string.
            Each term is pre-scaled and comes in size(batch, traj_len-1)
        """

        # NIMP needs
        if self._hyperparameters.loss_fn == LossFunction.NIMP:
            return self._loss_nimp(
                (
                    {k: v[..., 1:, :] for k, v in meas_contact_forces.items()}
                    if meas_contact_forces is not None
                    else None
                ),
                {k: v[..., 1:, :] for k, v in meas_contact_normals.items()},
                forward_args[2],  # estimated contact forces
                forward_args[3],  # estimated contact normals
                forward_args[4],  # estimated phi(t)
            )

        # VIMP needs timestamps, plant trajectory, and control
        return self._loss_vimp(
            (
                {k: v[..., 1:, :] for k, v in meas_contact_forces.items()}
                if meas_contact_forces is not None
                else None
            ),
            {k: v[..., 1:, :] for k, v in meas_contact_normals.items()},
            timestamps,
            forward_args[0],  # plant states
            forward_args[1],  # plant control
        )

    def observed_info(self, traj_data: TrajectorySet) -> Tensor:
        """Calculate Observed Information using ONLY partial derivatives

        Args:
            traj_data: Trajectory Data Collected

        Returns:
            Let n_params = len(current_learned_q) + len(self._multibody_terms.parameters())
            Returns Observed Info Matrix: (n_params, n_params)
        """

        # pylint: disable=too-many-locals, too-many-statements

        # TODO: generalize for >1 robot and object
        assert len(self._controlled_model_names) == 1
        assert len(self._learned_model_names) == 1

        print("Calculating Observed Info")
        print("Getting Current Pose Trajectory (no-diff)...")
        start = time.time()
        with torch.no_grad():
            timestamps = traj_data.get_full_trajectory(key="time")
            traj_len = timestamps.size()[0]
            plant_x, plant_u, _, _, _ = self.forward(
                ctrl_desired=traj_data.get_full_trajectory(
                    key=self.controlled_model_names[0] + "_desired"
                ),
                timestamps=timestamps,
                ctrl_actual=traj_data.get_full_trajectory(
                    key=self.controlled_model_names[0] + "_state"
                ),
            )
            state_param = Parameter(
                plant_x.detach().clone(),
                requires_grad=True,
            )
        print(f"... Done in {time.time() - start}s")

        print("Calculating per-timestep gradients...")
        start = time.time()
        n_geom = sum(
            param.numel()
            for param in self._multibody_terms.parameters()
            if param.requires_grad
        )
        n_params = n_geom + self.space.n_x
        outputs_phi = []
        outputs_normals = []

        def get_vjp(param_list, outputs, v):
            return torch.autograd.grad(
                outputs,
                param_list,
                v,
                create_graph=False,
                retain_graph=True,
            )

        def get_outputs_from_step_geom(
            step_dts: Tensor,
            step_x_batch: Tensor,
            plant_step_u: Tensor,
            geom_param_dict: dict[str, Tensor],
        ) -> Tensor:
            _, _, step_normals_batch, step_phis_batch = (
                self.forward_dynamics_functional(
                    self.space.q(step_x_batch),
                    self.space.v(step_x_batch),
                    plant_step_u,
                    step_dts,
                    geom_param_dict,
                    all_phis=False,
                    no_sim=True,
                )
            )
            output_phi_batch = torch.stack(list(step_phis_batch.values()), dim=-2)
            output_normals_batch = torch.stack(
                list(step_normals_batch.values()), dim=-2
            )

            return (
                output_phi_batch,
                output_normals_batch,
            )

        # Get outputs for each timestep
        print(f"Calc Outputs... ", end="")
        start = time.time()
        geom_param_dict = {
            k: Parameter(
                v.detach().unsqueeze(0).expand((traj_len,) + v.shape),
                requires_grad=True,
            )
            for k, v in self._multibody_terms.state_dict(keep_vars=True).items()
            if v.requires_grad == True
        }
        in_dims = (0, 0, 0, {k: 0 for k, _ in geom_param_dict.items()})
        # Repeat final DT for simulating to traj_len + 1
        step_dts = torch.cat(
            [timestamps[1:] - timestamps[:-1], timestamps[-1:] - timestamps[-2:-1]]
        )
        outputs_phi, outputs_normals = torch.vmap(
            get_outputs_from_step_geom,
            in_dims=in_dims,
            randomness="different",
        )(step_dts, state_param, plant_u.detach(), geom_param_dict)
        output_combined = torch.cat(
            [
                outputs_phi.reshape((traj_len, -1)),
                outputs_normals.reshape((traj_len, -1)),
            ],
            dim=-1,
        )
        n_outs = output_combined.shape[-1]
        jac_outs_params = torch.zeros((traj_len, n_outs, n_params))

        grad_outputs = []
        for n_out in range(n_outs):
            grad_out = torch.zeros_like(output_combined.T)
            grad_out[n_out, :] = torch.ones(traj_len)
            grad_outputs.append(grad_out)
        grad_outputs = torch.stack(grad_outputs)

        end = time.time() - start
        print(f"Done in {end:.3f}s")

        print(f"Calc Autodiff... ", end="")
        start = time.time()

        vjp_vmap = torch.vmap(
            partial(
                get_vjp,
                [state_param] + list(geom_param_dict.values()),
                output_combined.T,
            )
        )
        grads_ret = vjp_vmap(grad_outputs)
        jac_outs_params = torch.cat(
            [grad.reshape((n_outs, traj_len, -1)) for grad in grads_ret],
            dim=-1,
        ).transpose(0, 1)
        end = time.time() - start
        print(f"Done in {end:.3f}s")
        try:
            assert torch.all(~torch.isnan(jac_outs_params)), f"NaN in grads_combined"
        except AssertionError:
            breakpoint()
        # Clear the graph
        output_combined[0, 0].backward()

        # Switch to position parameter only
        n_params = n_geom + self._learned_trajectory.space.n_q
        jac_outs_params = torch.cat(
            [
                self.get_learned_trajectory(jac_outs_params[..., : self.space.n_x]),
                jac_outs_params[..., self.space.n_x :],
            ],
            dim=-1,
        )

        # Scale by Nominal Half Length
        # TODO: scale by geometry's nominal length instead
        jac_outs_params[..., -n_geom:] /= _NOMINAL_HALF_LENGTH

        print(f"... Done in {time.time() - start}s")
        print("Calculating info matrix...", end="")
        start = time.time()
        ret_info = torch.zeros((n_params, n_params))
        # Extract Contact Boolean
        phi_alpha = (
            np.log((1.0 / self._hyperparameters.w_phi_ci) - 1.0)
            / self._hyperparameters.w_phi_nominal
        )
        outputs_phi = outputs_phi.detach()
        outputs_normals = outputs_normals.detach()

        ### Phi Term
        grads_phi = jac_outs_params[1:, : outputs_phi[0].numel()].reshape(
            (traj_len - 1,) + outputs_phi.size()[1:] + (n_params,)
        )
        phi_mult = (
            phi_alpha
            * phi_alpha
            * torch.exp(phi_alpha * outputs_phi[1:])
            / torch.square(1.0 + torch.exp(phi_alpha * outputs_phi[1:]))
        ).unsqueeze(-1)
        info_phi = (
            pbmm(grads_phi.transpose(-1, -2), pbmm(phi_mult, grads_phi))
            .reshape((-1, n_params, n_params))
            .sum(dim=0)
        )
        ret_info += info_phi

        ### Normals Term
        contact_bool = torch.reciprocal(
            1.0 + torch.exp(phi_alpha * outputs_phi[1:])
        ).unsqueeze(-1)
        grads_normals = jac_outs_params[:-1, outputs_phi[0].numel() :].reshape(
            (traj_len - 1,) + outputs_normals.size()[1:] + (n_params,)
        )
        info_normals = (
            (
                contact_bool
                * (1.0 / self._hyperparameters.w_normal_var)
                * pbmm(grads_normals.transpose(-1, -2), grads_normals)
            )
            .reshape((-1, n_params, n_params))
            .sum(dim=0)
        )
        ret_info += info_normals
        print(f"...Done in {(time.time()-start):.6f}s")
        # print("Observed Info Breakpoint...")
        # breakpoint()
        return ret_info

    @torch.no_grad
    def learned_trajectory_rotate(
        self,
        quat_in: Optional[Tensor] = None,
    ) -> None:
        """Apply rotation to all poses in trajectory"""
        # 45 deg about +Z
        quat = (
            torch.tensor([0.9238795, 0.0, 0.0, 0.3826834])
            if quat_in is None
            else quat_in
        )  # 45deg about +Z
        assert quat.shape == (4,)
        quat_rotmat = Rotation.from_quat(
            quat.detach().cpu().numpy(), scalar_first=True
        ).as_matrix()

        current_traj = self._learned_trajectory.get_current_traj().detach().clone()
        assert current_traj.shape[-1] == 7
        current_rotmat = Rotation.from_quat(
            current_traj[..., :4].detach().cpu().numpy(), scalar_first=True
        ).as_matrix()
        new_rotmat = quat_rotmat @ current_rotmat

        current_traj[..., :4] = torch.tensor(
            Rotation.from_matrix(new_rotmat).as_quat(canonical=True, scalar_first=True)
        )
        self._learned_trajectory.overwrite_pose_params(current_traj)

    @torch.no_grad
    def learned_trajectory_average(
        self,
        quat_in: Optional[Tensor] = None,
    ) -> None:
        """Reset trajectory to average of current trajectory"""

        current_traj = self._learned_trajectory.get_current_traj().detach().clone()
        assert current_traj.shape[-1] == 7

        new_traj = current_traj.mean(dim=0).unsqueeze(0).expand(current_traj.shape)

        self._learned_trajectory.overwrite_pose_params(new_traj, set_v_to_zero=True)

    @torch.no_grad
    def learned_trajectory_sim_overwrite(
        self,
        traj_data: TrajectorySet,
    ) -> None:
        """Overwrite trajectory parameters with a simulation"""
        # TODO: generalize for >1 robot and object
        assert len(self._controlled_model_names) == 1
        assert len(self._learned_model_names) == 1
        timestamps = traj_data.get_full_trajectory(key="time")
        ctrl_desired = traj_data.get_full_trajectory(
            key=self.controlled_model_names[0] + "_desired"
        )
        try:
            plant_x, _, _, _, _ = self.diff_simulate(
                ctrl_desired,
                timestamps,
                learned_start_state=self._learned_trajectory.space.x(
                    self._learned_trajectory.current_pose_params(traj_num=0),
                    torch.zeros((self._learned_trajectory.space.n_v)),
                ),
            )
            obj_q = self.get_learned_trajectory(plant_x)
            self._learned_trajectory.overwrite_pose_params(obj_q)
        except diffcp.cone_program.SolverError:
            print("WARNING: Solver Errored, not overwriting parameters")

    def expected_fisher_info(
        self,
        ctrl_desired: Tensor,
        timestamps: Optional[Tensor] = None,
        current_learned_q: Optional[Tensor] = None,
    ) -> Tensor:
        """Calculate Expected Fisher Information

        Args:
            ctrl_desired: Input desired robot trajectory of size (batch, traj_len, self._controlled_space.n_x)
            timestamps: (traj_len,), if not provided use hyperparameters.default_dt
            current_learned_q: (self._learned_trajectory.space.n_q,), if not provided use current state

        Returns:
            Let n_params = len(current_learned_q) + len(self._multibody_terms.parameters())
            Returns Expected Fisher Info Matrices: (batch, n_params, n_params)
        """

        # pylint: disable=too-many-locals, too-many-statements
        self.zero_grad()
        print("Calculating Expected Information...")

        # TODO: generalize for >1 robot and object
        assert len(self._controlled_model_names) == 1
        assert len(self._learned_model_names) == 1

        # Input Validation
        full_batch_dims = ctrl_desired.size()[:-2]
        if len(full_batch_dims) < 1:
            full_batch_dims = (1,)
            ctrl_desired = ctrl_desired.unsqueeze(0)
        n_batches = math.prod(full_batch_dims)
        batch_dims = (n_batches,)
        ctrl_desired = ctrl_desired.reshape(batch_dims + ctrl_desired.size()[-2:])

        traj_len = ctrl_desired.size()[-2]
        assert ctrl_desired.size() == batch_dims + (
            traj_len,
            self._controlled_space.n_x,
        )
        timestamps = (
            torch.arange(
                traj_len * self._hyperparameters.default_dt,
                step=self._hyperparameters.default_dt,
            )
            if timestamps is None
            else timestamps
        )
        assert timestamps.size() == (traj_len,)
        pose_start = (
            self._learned_trajectory.current_pose_params(traj_num=-1)
            if current_learned_q is None
            else current_learned_q.detach().clone()
        )
        assert pose_start.size() == (self._learned_trajectory.space.n_q,)

        print("Getting Pose Trajectories (no-diff)...")
        ### PROFILING
        # import cProfile, pstats, io
        # from pstats import SortKey
        # pr = cProfile.Profile()
        # pr.enable()
        start = time.time()
        with torch.no_grad():
            plant_x_batch, plant_u_batch, _, _, _ = self.diff_simulate(
                ctrl_desired,
                timestamps,
                learned_start_state=self._learned_trajectory.space.x(
                    pose_start, torch.zeros((self._learned_trajectory.space.n_v))
                ),
            )
            try:
                assert torch.all(~torch.isnan(plant_x_batch)), f"NaN in plant_x_batch"
            except AssertionError:
                breakpoint()
            state_param_batch = Parameter(
                plant_x_batch.reshape((n_batches * traj_len, self.space.n_x)).clone(),
                requires_grad=True,
            )
        end = time.time()
        print(f"... Done in {end - start}s")
        ### PROFILING
        # s = io.StringIO()
        # sortby = SortKey.CUMULATIVE
        # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        # ps.print_stats(10)
        # print(s.getvalue())
        # breakpoint()

        print("Calculating per-timestep gradients...")
        n_geom = sum(
            param.numel()
            for param in self._multibody_terms.parameters()
            if param.requires_grad
        )
        n_params = n_geom + self.space.n_x

        def get_vjp(param_list, outputs, v):
            return torch.autograd.grad(
                outputs,
                param_list,
                v,
                create_graph=False,
                retain_graph=True,
            )

        def get_outputs_from_step_geom(
            step_dts: Tensor,
            step_x_batch: Tensor,
            plant_step_u: Tensor,
            geom_param_dict: dict[str, Tensor],
        ) -> Tensor:
            step_vplus_batch, step_forces_batch, step_normals_batch, step_phis_batch = (
                self.forward_dynamics_functional(
                    self.space.q(step_x_batch),
                    self.space.v(step_x_batch),
                    plant_step_u,
                    step_dts,
                    geom_param_dict,
                    all_phis=False,
                    no_sim=True,
                )
            )
            output_phi_batch = torch.cat(list(step_phis_batch.values()), dim=-2)
            output_normals_batch = torch.cat(list(step_normals_batch.values()), dim=-2)

            return (
                output_phi_batch,
                output_normals_batch,
            )

        # Get outputs for each timestep
        print(f"Calc Outputs... ", end="")
        start = time.time()
        geom_param_dict = {
            k: Parameter(
                v.detach().unsqueeze(0).expand((n_batches * traj_len,) + v.shape),
                requires_grad=True,
            )
            for k, v in self._multibody_terms.state_dict(keep_vars=True).items()
            if v.requires_grad == True
        }
        in_dims = (0, 0, 0, {k: 0 for k, _ in geom_param_dict.items()})
        # Repeat final DT for simulating to traj_len + 1
        step_dts = (
            torch.cat(
                [timestamps[1:] - timestamps[:-1], timestamps[-1:] - timestamps[-2:-1]]
            )
            .unsqueeze(0)
            .expand((n_batches, traj_len))
            .flatten()
        )
        outputs_phi, outputs_normals = torch.vmap(
            get_outputs_from_step_geom,
            in_dims=in_dims,
            randomness="different",
        )(
            step_dts,
            state_param_batch,
            plant_u_batch.reshape((n_batches * traj_len, -1)),
            geom_param_dict,
        )
        output_combined_batch = torch.cat(
            [
                outputs_phi.reshape((n_batches * traj_len, -1)),
                outputs_normals.reshape((n_batches * traj_len, -1)),
            ],
            dim=-1,
        )
        n_outs = output_combined_batch.shape[-1]
        jac_outs_params_batch = torch.zeros(batch_dims + (traj_len, n_outs, n_params))

        end = time.time() - start
        print(f"Done in {end:.3f}s")
        print(f"Calc Autodiff... ", end="")
        start = time.time()

        grad_outputs = []
        for n_out in range(n_outs):
            grad_out = torch.zeros_like(output_combined_batch.T)
            grad_out[n_out, :] = torch.ones(n_batches * traj_len)
            grad_outputs.append(grad_out)
        grad_outputs = torch.stack(grad_outputs)

        vjp_vmap = torch.vmap(
            partial(
                get_vjp,
                [state_param_batch] + list(geom_param_dict.values()),
                output_combined_batch.T,
            )
        )
        grads_ret_batch = vjp_vmap(grad_outputs)
        jac_outs_params_batch = (
            torch.cat(
                [
                    grad.reshape((n_outs, n_batches * traj_len, -1))
                    for grad in grads_ret_batch
                ],
                dim=-1,
            )
            .transpose(0, 1)
            .reshape((n_batches, traj_len, n_outs, -1))
        )
        end = time.time() - start
        print(f"Done in {end:.3f}s")
        try:
            assert torch.all(
                ~torch.isnan(jac_outs_params_batch)
            ), f"NaN in grads_combined"
        except AssertionError:
            breakpoint()

        # Clear the graph
        output_combined_batch[0, 0].backward()

        # Switch to position parameter only
        n_params = n_geom + self._learned_trajectory.space.n_q
        jac_outs_params_batch = torch.cat(
            [
                self.get_learned_trajectory(
                    jac_outs_params_batch[..., : self.space.n_x]
                ),
                jac_outs_params_batch[..., self.space.n_x :],
            ],
            dim=-1,
        )

        print("Calculating info matrix...")
        start = time.time()
        outputs_phi_batch = outputs_phi.detach().reshape(
            (
                n_batches,
                traj_len,
            )
            + outputs_phi.shape[1:]
        )
        outputs_normals_batch = outputs_normals.detach().reshape(
            (
                n_batches,
                traj_len,
            )
            + outputs_normals.shape[1:]
        )

        # TODO: consider returning a batch of gradients instead
        ret_info_batch = torch.zeros(batch_dims + (n_params, n_params))
        # Extract Contact Boolean
        phi_alpha = (
            np.log((1.0 / self._hyperparameters.w_phi_ci) - 1.0)
            / self._hyperparameters.w_phi_nominal
        )
        ### Phi Term
        n_contacts = outputs_phi_batch.size()[-2]
        grads_phi_batch = jac_outs_params_batch[..., 1:, :n_contacts, :].reshape(
            batch_dims + (traj_len - 1, n_contacts, 1, n_params)
        )
        phi_mult_batch = (
            phi_alpha
            * phi_alpha
            * torch.exp(phi_alpha * outputs_phi_batch[..., 1:, :, :])
            / torch.square(
                1.0 + torch.exp(phi_alpha * outputs_phi_batch[..., 1:, :, :])
            )
        ).unsqueeze(-1)
        # Nan -> inf/inf, but lim(outputs_phi -> inf) == 0
        phi_mult_batch[torch.isnan(phi_mult_batch)] = 0.0

        info_phi_batch = (
            pbmm(
                grads_phi_batch.transpose(-1, -2), pbmm(phi_mult_batch, grads_phi_batch)
            )
            .reshape(batch_dims + (-1, n_params, n_params))
            .sum(dim=-3)
        )
        ret_info_batch += info_phi_batch

        ### Normals Term
        contact_bool_batch = torch.reciprocal(
            1.0 + torch.exp(phi_alpha * outputs_phi_batch[..., 1:, :, :])
        ).unsqueeze(-1)
        grads_normals_batch = jac_outs_params_batch[..., :-1, n_contacts:, :].reshape(
            batch_dims + (traj_len - 1, n_contacts, 3, n_params)
        )
        info_normals_batch = (
            (
                contact_bool_batch
                * (1.0 / self._hyperparameters.w_normal_var)
                * pbmm(grads_normals_batch.transpose(-1, -2), grads_normals_batch)
            )
            .reshape(batch_dims + (-1, n_params, n_params))
            .sum(dim=-3)
        )
        ret_info_batch += info_normals_batch
        print(f"...Done in {(time.time()-start):.6f}s")
        # print("Expected Info Breakpoint...")
        # breakpoint()
        return ret_info_batch

    def observed_info_full(self, traj_data: TrajectorySet) -> Tensor:
        """Calculate Observed Information

        Args:
            traj_data: Trajectory Data Collected

        Returns:
            Let n_params = len(current_learned_q) + len(self._multibody_terms.parameters())
            Returns Observed Info Matrix: (n_params, n_params)
        """

        # pylint: disable=too-many-locals, too-many-statements

        # TODO: generalize for >1 robot and object
        assert len(self._controlled_model_names) == 1
        assert len(self._learned_model_names) == 1

        print("Calculating Observed Info")
        print("Getting Current Pose Trajectory (no-diff)...")
        start = time.time()
        with torch.no_grad():
            timestamps = traj_data.get_full_trajectory(key="time")
            traj_len = timestamps.size()[0]
            plant_x, plant_u, _, _, _ = self.forward(
                ctrl_desired=traj_data.get_full_trajectory(
                    key=self.controlled_model_names[0] + "_desired"
                ),
                timestamps=timestamps,
                ctrl_actual=traj_data.get_full_trajectory(
                    key=self.controlled_model_names[0] + "_state"
                ),
            )
            state_param = Parameter(
                plant_x.detach().clone(),
                requires_grad=True,
            )
        print(f"... Done in {time.time() - start}s")

        print("Calculating per-timestep gradients...")
        start = time.time()
        n_geom = sum(
            param.numel()
            for param in self._multibody_terms.parameters()
            if param.requires_grad
        )
        n_params = n_geom + self.space.n_x
        n_outs = -1  # Will be populated once the number of outputs is known
        jac_outs_params = None  # Will be populated once the number of outputs is known
        jac_xf_xn = (
            torch.eye(self.space.n_x)
            .reshape((1, self.space.n_x, self.space.n_x))
            .repeat((traj_len, 1, 1))
        )
        jac_partial_xnp_geom = torch.zeros((traj_len, self.space.n_x, n_geom))
        jac_xnp_xn = (
            torch.eye(self.space.n_x)
            .reshape((1, self.space.n_x, self.space.n_x))
            .repeat((traj_len, 1, 1))
        )
        outputs_phi = []
        outputs_normals = []
        outputs_forces = []

        def get_vjp(param_list, outputs, v):
            return torch.autograd.grad(
                outputs,
                param_list,
                v,
                create_graph=False,
                retain_graph=True,
            )

        def get_outputs_from_step_geom(
            step_dts: Tensor,
            step_x_batch: Tensor,
            plant_step_u: Tensor,
            geom_param_dict: dict[str, Tensor],
        ) -> Tensor:
            step_vplus_batch, step_forces_batch, step_normals_batch, step_phis_batch = (
                self.forward_dynamics_functional(
                    self.space.q(step_x_batch),
                    self.space.v(step_x_batch),
                    plant_step_u,
                    step_dts,
                    geom_param_dict,
                )
            )
            output_phi_batch = torch.stack(list(step_phis_batch.values()), dim=-2)
            output_normals_batch = torch.stack(
                list(step_normals_batch.values()), dim=-2
            )
            output_forces_batch = torch.stack(list(step_forces_batch.values()), dim=-2)
            output_x_batch = self.space.x(
                self.space.euler_step(
                    self.space.q(step_x_batch), step_vplus_batch, step_dts
                ),
                step_vplus_batch,
            )

            return (
                output_x_batch,
                output_phi_batch,
                output_forces_batch,
                output_normals_batch,
            )

        # Get outputs for each timestep
        print(f"Calc Outputs... ", end="")
        start = time.time()
        geom_param_dict = {
            k: Parameter(
                v.detach().unsqueeze(0).expand((traj_len,) + v.shape),
                requires_grad=True,
            )
            for k, v in self._multibody_terms.state_dict(keep_vars=True).items()
            if v.requires_grad == True
        }
        in_dims = (0, 0, 0, {k: 0 for k, _ in geom_param_dict.items()})
        # Repeat final DT for simulating to traj_len + 1
        step_dts = torch.cat(
            [timestamps[1:] - timestamps[:-1], timestamps[-1:] - timestamps[-2:-1]]
        )
        outputs_x, outputs_phi, outputs_forces, outputs_normals = torch.vmap(
            get_outputs_from_step_geom,
            in_dims=in_dims,
            randomness="different",
        )(step_dts, state_param, plant_u.detach(), geom_param_dict)
        output_combined = torch.cat(
            [
                outputs_x.reshape((traj_len, -1)),
                outputs_phi.reshape((traj_len, -1)),
                outputs_forces.reshape((traj_len, -1)),
                outputs_normals.reshape((traj_len, -1)),
            ],
            dim=-1,
        )
        n_outs = output_combined.shape[-1] - self.space.n_x
        if jac_outs_params is None:
            jac_outs_params = torch.zeros((traj_len, n_outs, n_params))

        grads_combined = torch.zeros((traj_len, self.space.n_x + n_outs, n_params))
        grad_outputs = []
        for n_out in range(self.space.n_x + n_outs):
            grad_out = torch.zeros_like(output_combined.T)
            grad_out[n_out, :] = torch.ones(traj_len)
            grad_outputs.append(grad_out)
        grad_outputs = torch.stack(grad_outputs)

        end = time.time() - start
        print(f"Done in {end:.3f}s")
        print(f"Calc Autodiff... ", end="")
        start = time.time()

        vjp_vmap = torch.vmap(
            partial(
                get_vjp,
                [state_param] + list(geom_param_dict.values()),
                output_combined.T,
            )
        )
        grads_ret = vjp_vmap(grad_outputs)
        grads_combined = torch.cat(
            [
                grad.reshape((self.space.n_x + n_outs, traj_len, -1))
                for grad in grads_ret
            ],
            dim=-1,
        ).transpose(0, 1)
        end = time.time() - start
        print(f"Done in {end:.3f}s")
        """
        try:
            assert torch.all(~torch.isnan(grads_combined)), f"NaN in grads_combined"
        except AssertionError:
            breakpoint()
        """
        # TODO: Hack away Nans
        grads_combined = torch.nan_to_num(grads_combined)
        # Clear the graph
        output_combined[0, 0].backward()

        print(f"Calc Jacobians... ", end="")
        start = time.time()
        for idx in range(traj_len):
            # Populate partial d(outputs)/d(x_n)
            jac_out_xn = grads_combined[
                idx, self.space.n_x :, : self.space.n_x
            ]  # n_phis x n_x
            jac_outs_params[idx, :, : self.space.n_x] = jac_out_xn

            # Populate partial d(outputs)/d(geom)
            jac_partial_out_geom = grads_combined[
                idx, self.space.n_x :, self.space.n_x :
            ]  # n_phis x n_geom
            jac_outs_params[idx, :, self.space.n_x :] = jac_partial_out_geom

            # No need to deal with x_{n+1} (past horizon)
            if idx < traj_len - 1:
                # Populate d(xn+1)/d(xt)
                jac_xnp_xn[idx, :, :] = grads_combined[
                    idx, : self.space.n_x, : self.space.n_x
                ]  # n_x x n_x

                # Update d(xH)/d(xt) forall t
                jac_xf_xn[: idx + 1, :, :] = (
                    jac_xnp_xn[idx, :, :].unsqueeze(-3) @ jac_xf_xn[: idx + 1, :, :]
                )

                # Populate partial d(x_[n+1])/d(geom)
                jac_partial_xnp_geom[..., idx, :, :] = grads_combined[
                    idx, : self.space.n_x, self.space.n_x :
                ]  # n_x x n_geom

            ### End loop: jac_outs_params combines jac_out_xn and jac_out_geom
            ###           also have all outputs

        assert torch.all(~torch.isnan(jac_outs_params)), "NaN in jac_outs_params"
        for idx in reversed(range(traj_len)):
            # Compute d(xn)/d(geom)
            jac_xn_geom = torch.zeros((self.space.n_x, n_geom))
            jac_xn2_xn = torch.eye(self.space.n_x)
            for idx2 in range(idx, traj_len - 1):
                jac_xn2_xn = jac_xnp_xn[idx2] @ jac_xn2_xn
                jac_partial_xn2_geom = jac_partial_xnp_geom[idx2]
                jac_xn_geom -= (
                    stable_inv(jac_xn2_xn, self._hyperparameters.rsim_eps)
                    @ jac_partial_xn2_geom
                )
            # Populate total derivative of jac_out_geom
            jac_partial_out_geom = jac_outs_params[idx, :, self.space.n_x :]
            jac_partial_out_xn = jac_outs_params[idx, :, : self.space.n_x]
            jac_out_geom = jac_partial_out_geom + jac_partial_out_xn @ jac_xn_geom
            jac_outs_params[idx, :, self.space.n_x :] = jac_out_geom
            breakpoint()

        ### Compute jac_out_xh instead of jac_out_xn
        for idx in reversed(range(traj_len)):
            jac_out_xn = jac_outs_params[idx, :, : self.space.n_x].clone()
            # jac_outs_xT @ jac_xf_xn = jac_outs_xn
            # jac_xf_xn.T @ jac_outs_xT.T = jac_outs_xn.T
            # jac_outs_xT.T = inv(jac_xf_xn.T) @ jac_outs_xn.T
            # pylint doesn't know about torch
            # pylint: disable-next=not-callable
            """
            jac_out_xh = torch.linalg.solve(
                (
                    torch.round(jac_xf_xn[idx, :, :], decimals=3).transpose(-1, -2)
                    + self._hyperparameters.rsim_eps * torch.eye(self.space.n_x)
                ),
                jac_out_xn.transpose(-1, -2),
            ).transpose(-1, -2)
            """
            jac_out_xh = jac_out_xn @ stable_inv(
                jac_xf_xn[idx].T, self._hyperparameters.rsim_eps
            )
            jac_outs_params[idx, :, : self.space.n_x] = jac_out_xh
            # breakpoint()

        # Switch to position parameter only
        n_params = n_geom + self._learned_trajectory.space.n_q
        jac_outs_params = torch.cat(
            [
                self.get_learned_trajectory(jac_outs_params[..., : self.space.n_x]),
                jac_outs_params[..., self.space.n_x :],
            ],
            dim=-1,
        )
        # TODO: Make Hyperparameter
        # Clamp to 1e6 for stability
        clamp_val = 1e8
        print(f"... Done in {time.time() - start}s")
        print("Calculating info matrix...", end="")
        start = time.time()
        ret_info = torch.zeros((n_params, n_params))
        # Extract Contact Boolean
        phi_alpha = (
            np.log((1.0 / self._hyperparameters.w_phi_ci) - 1.0)
            / self._hyperparameters.w_phi_nominal
        )
        outputs_phi = outputs_phi.detach()
        outputs_forces = outputs_forces.detach()
        outputs_normals = outputs_normals.detach()
        ### Phi Term
        grads_phi = torch.clamp(
            jac_outs_params[1:, : outputs_phi[0].numel()].reshape(
                (traj_len - 1,) + outputs_phi.size()[1:] + (n_params,)
            ),
            min=-clamp_val,
            max=clamp_val,
        )
        phi_mult = (
            phi_alpha
            * phi_alpha
            * torch.exp(phi_alpha * outputs_phi[1:])
            / torch.square(1.0 + torch.exp(phi_alpha * outputs_phi[1:]))
        ).unsqueeze(-1)
        info_phi = (
            pbmm(grads_phi.transpose(-1, -2), pbmm(phi_mult, grads_phi))
            .reshape((-1, n_params, n_params))
            .sum(dim=0)
        )
        ret_info += info_phi

        ### Forces Term
        grads_forces = jac_outs_params[
            :-1,
            outputs_phi[0].numel() : outputs_phi[0].numel() + outputs_forces[0].numel(),
        ].reshape((traj_len - 1,) + outputs_forces.size()[1:] + (n_params,))
        contact_bool = torch.reciprocal(
            1.0 + torch.exp(phi_alpha * outputs_phi[1:])
        ).unsqueeze(-1)
        info_forces = (
            (
                contact_bool
                * (1.0 / self._hyperparameters.w_force_var)
                * pbmm(grads_forces.transpose(-1, -2), grads_forces)
            )
            .reshape((-1, n_params, n_params))
            .sum(dim=0)
        )
        # TODO: make argument, remove forces from info
        # ret_info += info_forces

        ### Normals Term
        grads_normals = torch.clamp(
            jac_outs_params[
                :-1, outputs_phi[0].numel() + outputs_forces[0].numel() :
            ].reshape((traj_len - 1,) + outputs_normals.size()[1:] + (n_params,)),
            min=-clamp_val,
            max=clamp_val,
        )
        info_normals = (
            (
                contact_bool
                * (1.0 / self._hyperparameters.w_normal_var)
                * pbmm(grads_normals.transpose(-1, -2), grads_normals)
            )
            .reshape((-1, n_params, n_params))
            .sum(dim=0)
        )
        ret_info += info_normals
        print(f"...Done in {(time.time()-start):.6f}s")
        breakpoint()
        return ret_info

    def expected_fisher_info_full(
        self,
        ctrl_desired: Tensor,
        timestamps: Optional[Tensor] = None,
        current_learned_q: Optional[Tensor] = None,
    ) -> Tensor:
        """Calculate Expected Fisher Information

        Args:
            ctrl_desired: Input desired robot trajectory of size (batch, traj_len, self._controlled_space.n_x)
            timestamps: (traj_len,), if not provided use hyperparameters.default_dt
            current_learned_q: (self._learned_trajectory.space.n_q,), if not provided use current state

        Returns:
            Let n_params = len(current_learned_q) + len(self._multibody_terms.parameters())
            Returns Expected Fisher Info Matrices: (batch, n_params, n_params)
        """

        # pylint: disable=too-many-locals, too-many-statements
        self.zero_grad()
        print("Calculating Expected Information...")

        # TODO: generalize for >1 robot and object
        assert len(self._controlled_model_names) == 1
        assert len(self._learned_model_names) == 1

        # Input Validation
        full_batch_dims = ctrl_desired.size()[:-2]
        if len(full_batch_dims) < 1:
            full_batch_dims = (1,)
            ctrl_desired = ctrl_desired.unsqueeze(0)
        n_batches = math.prod(full_batch_dims)
        batch_dims = (n_batches,)
        ctrl_desired = ctrl_desired.reshape(batch_dims + ctrl_desired.size()[-2:])

        traj_len = ctrl_desired.size()[-2]
        assert ctrl_desired.size() == batch_dims + (
            traj_len,
            self._controlled_space.n_x,
        )
        timestamps = (
            torch.arange(
                traj_len * self._hyperparameters.default_dt,
                step=self._hyperparameters.default_dt,
            )
            if timestamps is None
            else timestamps
        )
        assert timestamps.size() == (traj_len,)
        pose_start = (
            self._learned_trajectory.current_pose_params(traj_num=-1)
            if current_learned_q is None
            else current_learned_q.detach().clone()
        )
        assert pose_start.size() == (self._learned_trajectory.space.n_q,)

        print("Getting Pose Trajectories (no-diff)...")
        start = time.time()
        with torch.no_grad():
            plant_x_batch, plant_u_batch, _, _, _ = self.diff_simulate(
                ctrl_desired,
                timestamps,
                learned_start_state=self._learned_trajectory.space.x(
                    pose_start, torch.zeros((self._learned_trajectory.space.n_v))
                ),
            )
            try:
                assert torch.all(~torch.isnan(plant_x_batch)), f"NaN in plant_x_batch"
            except AssertionError:
                breakpoint()
            state_param_batch = Parameter(
                plant_x_batch.reshape((n_batches * traj_len, self.space.n_x)).clone(),
                requires_grad=True,
            )
        print(f"... Done in {time.time() - start}s")

        print("Calculating per-timestep gradients...")
        n_geom = sum(
            param.numel()
            for param in self._multibody_terms.parameters()
            if param.requires_grad
        )
        n_params = n_geom + self.space.n_x
        n_outs = -1  # Will be populated once the number of outputs is known
        jac_outs_params_batch = (
            None  # Will be populated once the number of outputs is known
        )
        jac_xf_xn_batch = (
            torch.eye(self.space.n_x)
            .reshape((1,) * len(batch_dims) + (1, self.space.n_x, self.space.n_x))
            .repeat(batch_dims + (traj_len, 1, 1))
        )
        jac_partial_xnp_geom_batch = torch.zeros(
            batch_dims + (traj_len, self.space.n_x, n_geom)
        )
        jac_xnp_xn_batch = (
            torch.eye(self.space.n_x)
            .reshape((1,) * len(batch_dims) + (1, self.space.n_x, self.space.n_x))
            .repeat(batch_dims + (traj_len, 1, 1))
        )
        jac_xnp_xz_batch = (
            torch.eye(self.space.n_x)
            .reshape((1,) * len(batch_dims) + (1, self.space.n_x, self.space.n_x))
            .repeat(batch_dims + (traj_len, 1, 1))
        )

        def get_vjp(param_list, outputs, v):
            return torch.autograd.grad(
                outputs,
                param_list,
                v,
                create_graph=False,
                retain_graph=True,
            )

        def get_outputs_from_step_geom(
            step_dts: Tensor,
            step_x_batch: Tensor,
            plant_step_u: Tensor,
            geom_param_dict: dict[str, Tensor],
        ) -> Tensor:
            step_vplus_batch, step_forces_batch, step_normals_batch, step_phis_batch = (
                self.forward_dynamics_functional(
                    self.space.q(step_x_batch),
                    self.space.v(step_x_batch),
                    plant_step_u,
                    step_dts,
                    geom_param_dict,
                )
            )
            output_phi_batch = torch.cat(list(step_phis_batch.values()), dim=-2)
            output_normals_batch = torch.cat(list(step_normals_batch.values()), dim=-2)
            output_forces_batch = torch.cat(list(step_forces_batch.values()), dim=-2)
            output_x_batch = self.space.x(
                self.space.euler_step(
                    self.space.q(step_x_batch), step_vplus_batch, step_dts
                ),
                step_vplus_batch,
            )

            return (
                output_x_batch,
                output_phi_batch,
                output_forces_batch,
                output_normals_batch,
            )

        # Get outputs for each timestep
        print(f"Calc Outputs... ", end="")
        start = time.time()
        geom_param_dict = {
            k: Parameter(
                v.detach().unsqueeze(0).expand((n_batches * traj_len,) + v.shape),
                requires_grad=True,
            )
            for k, v in self._multibody_terms.state_dict(keep_vars=True).items()
            if v.requires_grad == True
        }
        in_dims = (0, 0, 0, {k: 0 for k, _ in geom_param_dict.items()})
        # Repeat final DT for simulating to traj_len + 1
        step_dts = (
            torch.cat(
                [timestamps[1:] - timestamps[:-1], timestamps[-1:] - timestamps[-2:-1]]
            )
            .unsqueeze(0)
            .expand((n_batches, traj_len))
            .flatten()
        )
        outputs_x, outputs_phi, outputs_forces, outputs_normals = torch.vmap(
            get_outputs_from_step_geom,
            in_dims=in_dims,
            randomness="different",
        )(
            step_dts,
            state_param_batch,
            plant_u_batch.reshape((n_batches * traj_len, -1)),
            geom_param_dict,
        )
        output_combined_batch = torch.cat(
            [
                outputs_x.reshape((n_batches * traj_len, -1)),
                outputs_phi.reshape((n_batches * traj_len, -1)),
                outputs_forces.reshape((n_batches * traj_len, -1)),
                outputs_normals.reshape((n_batches * traj_len, -1)),
            ],
            dim=-1,
        )
        n_outs = output_combined_batch.shape[-1] - self.space.n_x
        if jac_outs_params_batch is None:
            jac_outs_params_batch = torch.zeros(
                batch_dims + (traj_len, n_outs, n_params)
            )

        end = time.time() - start
        print(f"Done in {end:.3f}s")
        print(f"Calc Autodiff... ", end="")
        start = time.time()

        grads_combined_batch = torch.zeros(
            batch_dims + (traj_len, self.space.n_x + n_outs, n_params)
        )
        grad_outputs = []
        for n_out in range(self.space.n_x + n_outs):
            grad_out = torch.zeros_like(output_combined_batch.T)
            grad_out[n_out, :] = torch.ones(n_batches * traj_len)
            grad_outputs.append(grad_out)
        grad_outputs = torch.stack(grad_outputs)

        vjp_vmap = torch.vmap(
            partial(
                get_vjp,
                [state_param_batch] + list(geom_param_dict.values()),
                output_combined_batch.T,
            )
        )
        grads_ret_batch = vjp_vmap(grad_outputs)
        grads_combined_batch = (
            torch.cat(
                [
                    grad.reshape((self.space.n_x + n_outs, n_batches * traj_len, -1))
                    for grad in grads_ret_batch
                ],
                dim=-1,
            )
            .transpose(0, 1)
            .reshape((n_batches, traj_len, self.space.n_x + n_outs, -1))
        )
        end = time.time() - start
        print(f"Done in {end:.3f}s")
        """
        try:
            assert torch.all(~torch.isnan(grads_combined_batch)), f"NaN in grads_combined"
        except AssertionError:
            breakpoint()
        """
        # TODO: HACK away NaNs
        grads_combined_batch = torch.nan_to_num(grads_combined_batch)
        # Clear the graph
        output_combined_batch[0, 0].backward()

        for idx in range(traj_len):
            # Populate partial d(outputs)/d(x_n)
            jac_out_xn_batch = grads_combined_batch[
                ..., idx, self.space.n_x :, : self.space.n_x
            ]  # n_outs x self.space.n_x
            jac_outs_params_batch[..., idx, :, : self.space.n_x] = jac_out_xn_batch

            # Populate partial d(outputs)/d(geom)
            jac_partial_out_geom_batch = grads_combined_batch[
                ..., idx, self.space.n_x :, self.space.n_x :
            ]  # n_phis x n_geom
            jac_outs_params_batch[..., idx, :, self.space.n_x :] = (
                jac_partial_out_geom_batch
            )

            # No need to deal with x_{n+1} (past horizon)
            if idx < traj_len - 1:
                # Update d(xT)/d(xt) for all previous timesteps to be d(xn+1)/d(xt)
                jac_xnp_xn_batch[..., idx, :, :] = grads_combined_batch[
                    ..., idx, : self.space.n_x, : self.space.n_x
                ]  # n_x x n_x

                jac_xf_xn_batch[..., : idx + 1, :, :] = (
                    jac_xnp_xn_batch[..., idx, :, :].unsqueeze(-3)
                    @ jac_xf_xn_batch[..., : idx + 1, :, :]
                )

                jac_xn_xz_batch = jac_xnp_xz_batch[..., idx:, :, :]
                jac_xnp_xz_batch[..., idx:, :, :] = (
                    jac_xnp_xn_batch[..., idx, :, :].unsqueeze(-3) @ jac_xn_xz_batch
                )

                # Populate partial d(x_[n+1])/d(geom)
                jac_partial_xnp_geom_batch[..., idx, :, :] = grads_combined_batch[
                    ..., idx, : self.space.n_x, self.space.n_x :
                ]  # n_x x n_geom

            ### End loop: jac_outs_params_batch combines jac_out_xn and jac_out_geom
            ###           also have all outputs
        assert torch.all(
            ~torch.isnan(jac_outs_params_batch)
        ), "NaN in jac_outs_params_batch"

        ### Compute jac_xz_geom_batch
        jac_xz_geom_batch = torch.zeros(batch_dims + (self.space.n_x, n_geom))
        for idx in range(traj_len):
            # pylint doesn't know about torch
            # pylint: disable-next=not-callable
            jac_xz_geom_batch -= torch.linalg.solve(
                (
                    torch.round(jac_xnp_xz_batch[..., idx, :, :], decimals=3)
                    + self._hyperparameters.rsim_eps * torch.eye(self.space.n_x)
                ),
                jac_partial_xnp_geom_batch[..., idx, :, :],
            )

        ### Compute jac_out_geom total instead of partial
        jac_xn_geom_batch = jac_xz_geom_batch.clone()
        for idx in range(traj_len):
            jac_out_xn_batch = jac_outs_params_batch[..., idx, :, : self.space.n_x]
            jac_partial_out_geom_batch = jac_outs_params_batch[
                ..., idx, :, self.space.n_x :
            ]

            jac_outs_params_batch[..., idx, :, self.space.n_x :] = (
                jac_partial_out_geom_batch + jac_out_xn_batch @ jac_xn_geom_batch
            )

            if idx < traj_len - 1:
                jac_xn_geom_batch = (
                    jac_partial_xnp_geom_batch[..., idx, :, :]
                    + jac_xnp_xn_batch[..., idx, :, :] @ jac_xn_geom_batch
                )

        ### Computer jac_out_xh instead of jac_out_xn
        for idx in range(traj_len):
            jac_out_xn_batch = jac_outs_params_batch[..., idx, :, : self.space.n_x]
            # jac_outs_xT @ jac_xf_xn = jac_outs_xn
            # jac_xf_xn.T @ jac_outs_xT.T = jac_outs_xn.T
            # jac_outs_xT.T = inv(jac_xf_xn.T) @ jac_outs_xn.T
            # pylint doesn't know about torch
            # pylint: disable-next=not-callable
            jac_out_xh_batch = torch.linalg.solve(
                (
                    torch.round(jac_xf_xn_batch[..., idx, :, :], decimals=3).transpose(
                        -1, -2
                    )
                    + self._hyperparameters.rsim_eps * torch.eye(self.space.n_x)
                ),
                jac_out_xn_batch.transpose(-1, -2),
            ).transpose(-1, -2)
            jac_outs_params_batch[..., idx, :, : self.space.n_x] = jac_out_xh_batch

        # Switch to position parameter only
        n_params = n_geom + self._learned_trajectory.space.n_q
        jac_outs_params_batch = torch.cat(
            [
                self.get_learned_trajectory(
                    jac_outs_params_batch[..., : self.space.n_x]
                ),
                jac_outs_params_batch[..., self.space.n_x :],
            ],
            dim=-1,
        )
        # TODO: Make Hyperparameter
        # Clamp to 1e6 for stability
        clamp_val = 1e2
        print("Calculating info matrix...")
        start = time.time()
        outputs_phi_batch = outputs_phi.detach().reshape(
            (
                n_batches,
                traj_len,
            )
            + outputs_phi.shape[1:]
        )
        outputs_normals_batch = outputs_normals.detach().reshape(
            (
                n_batches,
                traj_len,
            )
            + outputs_normals.shape[1:]
        )
        outputs_forces_batch = outputs_forces.detach().reshape(
            (
                n_batches,
                traj_len,
            )
            + outputs_forces.shape[1:]
        )
        # TODO: consider returning a batch of gradients instead
        ret_info_batch = torch.zeros(batch_dims + (n_params, n_params))
        # Extract Contact Boolean
        phi_alpha = (
            np.log((1.0 / self._hyperparameters.w_phi_ci) - 1.0)
            / self._hyperparameters.w_phi_nominal
        )
        ### Phi Term
        n_contacts = outputs_phi_batch.size()[-2]
        grads_phi_batch = torch.clamp(
            jac_outs_params_batch[..., 1:, :n_contacts, :].reshape(
                batch_dims + (traj_len - 1, n_contacts, 1, n_params)
            ),
            min=-clamp_val,
            max=clamp_val,
        )
        phi_mult_batch = (
            phi_alpha
            * phi_alpha
            * torch.exp(phi_alpha * outputs_phi_batch[..., 1:, :, :])
            / torch.square(
                1.0 + torch.exp(phi_alpha * outputs_phi_batch[..., 1:, :, :])
            )
        ).unsqueeze(-1)
        # Nan -> inf/inf, but lim(outputs_phi -> inf) == 0
        phi_mult_batch[torch.isnan(phi_mult_batch)] = 0.0

        info_phi_batch = (
            pbmm(
                grads_phi_batch.transpose(-1, -2), pbmm(phi_mult_batch, grads_phi_batch)
            )
            .reshape(batch_dims + (-1, n_params, n_params))
            .sum(dim=-3)
        )
        ret_info_batch += info_phi_batch

        ### Forces Term
        n_forces = n_contacts * 3
        grads_forces_batch = jac_outs_params_batch[
            ..., :-1, n_contacts : n_contacts + n_forces, :
        ].reshape(batch_dims + (traj_len - 1, n_contacts, 3, n_params))
        contact_bool_batch = torch.reciprocal(
            1.0 + torch.exp(phi_alpha * outputs_phi_batch[..., 1:, :, :])
        ).unsqueeze(-1)
        info_forces_batch = (
            (
                contact_bool_batch
                * (1.0 / self._hyperparameters.w_force_var)
                * pbmm(grads_forces_batch.transpose(-1, -2), grads_forces_batch)
            )
            .reshape(batch_dims + (-1, n_params, n_params))
            .sum(dim=-3)
        )
        # TODO: make argument to remove forces from info
        # ret_info_batch += info_forces_batch

        ### Normals Term
        grads_normals_batch = torch.clamp(
            jac_outs_params_batch[..., :-1, n_contacts + n_forces :, :].reshape(
                batch_dims + (traj_len - 1, n_contacts, 3, n_params)
            ),
            min=-clamp_val,
            max=clamp_val,
        )
        info_normals_batch = (
            (
                contact_bool_batch
                * (1.0 / self._hyperparameters.w_normal_var)
                * pbmm(grads_normals_batch.transpose(-1, -2), grads_normals_batch)
            )
            .reshape(batch_dims + (-1, n_params, n_params))
            .sum(dim=-3)
        )
        ret_info_batch += info_normals_batch
        print(f"...Done in {(time.time()-start):.6f}s")
        return ret_info_batch

    @torch.no_grad
    def get_learned_body_name(self) -> str:
        """Name of learned body (assumed for contact)."""
        assert len(self._learned_model_names) == 1, "Only 1 learnable object supported"
        model_name = self._learned_model_names[0]
        plant = self._multibody_terms.plant_diagram.plant
        bodies = get_bodies_in_model_instance(
            plant, plant.GetModelInstanceByName(model_name)
        )
        assert len(bodies) == 1, "Only 1 learnable body supported"
        return bodies[0].name()

    @torch.no_grad
    def get_learned_geometry(self, surface_sample=False, sample_count=1000) -> Shape:
        """Current geometry as a Drake Shape."""
        assert len(self._learned_model_names) == 1, "Only 1 learnable object supported"
        model_name = self._learned_model_names[0]
        plant = self._multibody_terms.plant_diagram.plant
        bodies = get_bodies_in_model_instance(
            plant, plant.GetModelInstanceByName(model_name)
        )
        assert len(bodies) == 1, "Only 1 learnable body supported"
        body_id = unique_body_identifier(plant, bodies[0])
        body_geometry_indices = self._multibody_terms.geometry_body_assignment[body_id]
        assert len(body_geometry_indices) == 1, "Only 1 learnable geometry"
        body_geometry = cast(
            CollisionGeometry,
            self._multibody_terms.contact_terms.geometries[body_geometry_indices[0]],
        )
        if surface_sample:
            return body_geometry.sample_surface(sample_count)
        return PydrakeToCollisionGeometryFactory.reverse_convert(body_geometry)

    @torch.no_grad
    def get_learned_centroid(self) -> np.ndarray:
        """Current geometric centroid for learned object"""
        pose = self.get_learned_pose().detach().cpu().numpy()
        shape = self.get_learned_geometry()
        if isinstance(shape, pydrake.geometry.Mesh) or isinstance(
            shape, pydrake.geometry.Convex
        ):
            centroid = shape.GetConvexHull().centroid()
            pose[4:] = (
                Rotation.from_quat(pose[:4], scalar_first=True).apply(centroid)
                + pose[4:]
            )
        return pose

    @torch.no_grad
    def get_body_geometry(
        self, body_name: str, surface_sample=False, sample_count=500
    ) -> Union[Shape, np.ndarray]:
        """Current geometry of body based on name"""
        plant = self._multibody_terms.plant_diagram.plant
        body = plant.GetBodyByName(body_name)
        body_id = unique_body_identifier(plant, body)
        body_geometry_indices = self._multibody_terms.geometry_body_assignment[body_id]
        assert len(body_geometry_indices) == 1, "Body must contain only 1 geometry"
        body_geometry = cast(
            CollisionGeometry,
            self._multibody_terms.contact_terms.geometries[body_geometry_indices[0]],
        )
        if surface_sample:
            return body_geometry.sample_surface(sample_count)
        return PydrakeToCollisionGeometryFactory.reverse_convert(body_geometry)

    @torch.no_grad
    def get_learned_pose(self) -> Tensor:
        """Current pose for the learned object"""
        return (
            self._learned_trajectory.current_pose_params(traj_num=-1).detach().clone()
        )

    @torch.no_grad
    def get_learned_trajectory(self, system_traj: Optional[Tensor] = None) -> Tensor:
        """Current pose trajectory for the learned object

        If provided, extract full trajectory from the system trajectory.

        Args:
        system_traj : (batch, traj_len, self.space.n_x)
        """
        if system_traj is not None:
            data_state = self._multibody_terms.model_states_from_state_tensor(
                system_traj
            )
            ret_list = []
            for traj_model_idx, traj_model_name in enumerate(self._learned_model_names):
                ret_list.append(
                    self._learned_trajectory.space.spaces[traj_model_idx].q(
                        data_state[traj_model_name + "_state"]
                    )
                )
            ret = torch.cat(ret_list, dim=-1)
            assert ret.size() == system_traj.size()[:-1] + (
                self._learned_trajectory.space.n_q,
            ), ret.size()
            return ret
        return self._learned_trajectory.get_current_traj()

    @torch.no_grad
    def get_controlled_trajectory(self, system_traj: Tensor) -> Tensor:
        """Current pose trajectory for the robot object

        Extract full trajectory from the system trajectory.

        Args:
        system_traj : (batch, traj_len, self.space.n_x)
        """
        assert system_traj is not None
        data_state = self._multibody_terms.model_states_from_state_tensor(system_traj)
        ret_list = []
        for traj_model_idx, traj_model_name in enumerate(self._controlled_model_names):
            ret_list.append(
                self._controlled_space.spaces[traj_model_idx].q(
                    data_state[traj_model_name + "_state"]
                )
            )
        ret = torch.cat(ret_list, dim=-1)
        assert ret.size() == system_traj.size()[:-1] + (
            self._controlled_space.n_q,
        ), ret.size()
        return ret
