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
from typing import override, Optional, cast

import gin
import numpy as np
from pydrake.geometry import Shape
from tensordict import TensorDict
import torch
from torch import Tensor
from torch.nn import Module

from dair_pll.drake_utils import (
    unique_body_identifier,
    get_bodies_in_model_instance,
)
from dair_pll.geometry import CollisionGeometry, PydrakeToCollisionGeometryFactory
from dair_pll.learnable_trajectory import LearnableTrajectories
from dair_pll.multibody_terms import MultibodyTerms, LearnableBodySettings
from dair_pll.solvers import DynamicCvxpyLCQPLayer
from dair_pll.state_space import StateSpace, ProductSpace
from dair_pll.tensor_utils import pbmm, broadcast_lorentz, sappy_reorder_mat


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
        self._solver = DynamicCvxpyLCQPLayer()
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

    def forward_dynamics(
        self,
        step_q: Tensor,
        step_v: Tensor,
        step_u: Tensor,
        step_dt: Optional[float | Tensor] = None,
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
        assert step_q.size() == batch_dims + (
            self._multibody_terms.plant_diagram.space.n_q,
        )
        assert step_v.size() == batch_dims + (
            self._multibody_terms.plant_diagram.space.n_v,
        )
        assert step_u.size() == batch_dims + (self._controlled_space.n_v,)
        dt = self._hyperparameters.default_dt if step_dt is None else step_dt
        phi_eps = 1e-2
        eps = torch.finfo(step_q.dtype).eps
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

        reorder_mat = sappy_reorder_mat(n_contacts)
        reorder_mat = reorder_mat.reshape(
            (1,) * (m_delassus.dim() - 2) + reorder_mat.shape
        ).expand(m_delassus.shape)

        mq_delassus = m_delassus + eps * torch.eye(3 * n_contacts)

        # pylint: disable=E1103
        double_zero_vector = torch.zeros(m_phi.shape[:-1] + (2 * n_contacts,))
        phi_then_zero = torch.cat((m_phi, double_zero_vector), dim=-1).unsqueeze(-1)

        step_v_minus = step_v + dt * non_contact_acceleration
        q_full = pbmm(m_jac, step_v_minus.unsqueeze(-1)) + (1 / dt) * phi_then_zero

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
                ret_phis[key] = torch.ones(batch_dims + (1, 1)) * m_phi[..., index]

        # Trajectory Jumps, return 0s
        if dt > self._hyperparameters.dt_thresh:
            # print("Trajectory Jump, Assuming No Contact")
            return (
                torch.zeros_like(step_v),
                ret_contact_forces,
                ret_contact_normals,
                ret_phis,
            )

        ## Solve Impulase
        impulse_full = pbmm(
            reorder_mat,
            self._solver(
                pbmm(
                    reorder_mat.transpose(-1, -2), pbmm(mq_delassus, reorder_mat)
                ),  # Quadratic Term
                pbmm(reorder_mat.transpose(-1, -2), q_full).squeeze(-1),  # Linear Term
            ).unsqueeze(-1),
        )

        impulse = torch.zeros_like(impulse_full)
        impulse[contact_filter] += impulse_full[contact_filter]

        # pylint doesn't know about torch
        # pylint: disable=not-callable
        step_v_add = torch.linalg.solve(
            m_mass, pbmm(m_jac.transpose(-1, -2), impulse)
        ).squeeze(-1)

        ### Populate contact forces / normals
        for key, ret_contact_force in ret_contact_forces.items():
            indices = np.array([i for i, x in enumerate(obj_pair_list) if x == key])
            if len(indices) == 0:
                continue
            assert len(indices) == 1
            index = indices[0]
            fric_index = len(obj_pair_list) + 2 * index
            mr_wf_i = mr_fw_list[index].transpose(-1, -2).detach()
            ret_contact_normals[key][contact_filter[..., index, 0], 0, :] = mr_wf_i[
                contact_filter[..., index, 0], :, 2
            ]
            # Force in contact_frame
            ret_contact_force[..., 0, 2] = impulse[..., index, 0] / dt
            ret_contact_force[..., 0, :2] = (
                mu_list[index].detach()
                * impulse[..., fric_index : fric_index + 2, 0]
                / dt
            )
            # Rotate into world frame
            ret_contact_forces[key] = pbmm(
                mr_wf_i, ret_contact_force.transpose(-1, -2)
            ).transpose(-1, -2)
        ###
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
        ctrl_actual: Optional[Tensor],
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
        Returns:
            - state tensor of size (batch, traj_len, plant.n_x)
            - robot u  (batch, traj_len, robot.n_v)
            - ret_contact_forces Dict[<collision>, size(batch, traj_len, 3)]
            - ret_normal_forces Dict[<collision>, size(batch, traj_len, 3)]
            - ret_phis Dict[<collision>, size(batch, traj_len, 1)]
        """
        # pylint: disable=too-many-locals, too-many-branches

        # Input Validation
        assert len(ctrl_desired.size()) >= 2
        batch_dims = ctrl_desired.size()[:-2]
        traj_len = ctrl_desired.size()[-2]
        assert ctrl_desired.size() == batch_dims + (
            traj_len,
            self._controlled_space.n_x,
        ), str(ctrl_desired.size())
        assert timestamps.size() == (traj_len,)

        assert (ctrl_actual is None) or (
            ctrl_actual.size()
            == batch_dims
            + (
                traj_len,
                self._controlled_space.n_x,
            )
        )

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
                self._learned_trajectory.space.x_split(
                    self._learned_trajectory.current_state()
                ),
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
            step_q, step_v = self._multibody_terms.plant_diagram.space.q_v(
                self._multibody_terms.construct_state_tensor(
                    data_state[..., sim_idx - 1]
                )
            )
            # print(f"With robot_state: {data_state[..., sim_idx - 1]["robot_state"].detach().cpu().numpy()}")
            # print(f"With cube_state: {data_state[..., sim_idx - 1]["cube_state"].detach().cpu().numpy()}")
            step_vplus, step_contact_forces, step_contact_normals, step_contact_phis = (
                self.forward_dynamics(step_q, step_v, torch.cat(step_u, dim=-1), sim_dt)
            )
            data_state[..., sim_idx] = (
                self._multibody_terms.model_states_from_state_tensor(
                    self._multibody_terms.plant_diagram.space.x(
                        self._multibody_terms.plant_diagram.space.euler_step(
                            step_q, step_vplus, sim_dt
                        ),
                        step_vplus,
                    )
                )
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
        step_q, step_v = self._multibody_terms.plant_diagram.space.q_v(
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
            ret_contact_phis[key][..., traj_len - 1, :] = final_phi[..., indices[0]]

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
        assert timestamps.size() == (traj_len,)

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
        meas_contact_forces: dict[tuple[str, str], Tensor],
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
            assert meas_contact_forces[key].size() == batch_dims + (
                traj_len - 1,
                3,
            ), meas_contact_forces[key].size()

        ret_loss = {
            "loss_meas_bool": torch.zeros(batch_dims + (traj_len - 1,)),
            "loss_meas_force": torch.zeros(batch_dims + (traj_len - 1,)),
            "loss_meas_normal": torch.zeros(batch_dims + (traj_len - 1,)),
        }

        # Supervise each key
        for key in supervised_keys:
            # Input Validation
            assert key in meas_contact_forces, f"Key {key} not in meas_contact_forces"
            assert key in meas_contact_normals, f"Key {key} not in meas_contact_normals"
            assert key in est_contact_forces, f"Key {key} not in est_contact_forces"
            assert key in est_contact_normals, f"Key {key} not in est_contact_normals"
            assert key in est_contact_phis, f"Key {key} not in est_contact_phis"
            contact_bool = torch.ones(batch_dims + (traj_len - 1,))
            contact_bool[
                torch.isclose(
                    # pylint doesn't know about torch
                    # pylint: disable=not-callable
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

            loss_meas_force = (
                0.5
                * (
                    torch.ones_like(contact_bool)
                    if self._hyperparameters.supervise_non_contact_force
                    else contact_bool
                )
                * (1.0 / self._hyperparameters.w_force_var)
                * pbmm(
                    (est_contact_forces[key] - meas_contact_forces[key]).unsqueeze(-2),
                    (est_contact_forces[key] - meas_contact_forces[key]).unsqueeze(-1),
                )
                .squeeze(-2)
                .squeeze(-1)
            )
            assert loss_meas_force.size() == batch_dims + (
                traj_len - 1,
            ), loss_meas_force.size()
            ret_loss["loss_meas_force"] += loss_meas_force

        return ret_loss

    def _loss_vimp(
        self,
        meas_contact_forces,
        meas_contact_normals,
        est_contact_forces,
        est_contact_normals,
        est_contact_phis,
    ) -> dict[str, Tensor]:
        """
        VIMP Loss
        contact_normal = 0 means that no contact is detected.
        if len(meas_contact_forces/normals) > traj_len-1 (could be traj_len),
            only use the first traj_len-1

        Args:
            meas_contact_forces: Dict[<collision>, size(batch, traj_len-1, 3)] in world frame
            meas_contact_normals: Dict[<collision>, size(batch, traj_len-1, 3)] in world frame
            est_contact_forces: Dict[<collision>, size(batch, traj_len-1, 3)] in world frame
            est_contact_normals: Dict[<collision>, size(batch, traj_len-1, 3)] in world frame
            est_contact_phis: Dict[<collision>, size(batch, traj_len, 1)] in world frame
        Returns:
            Dictionary of loss terms, each identified by a string.
            Each term is pre-scaled and comes in size(batch, traj_len-1)
            Loss Terms Are:
             * Contact Boolean Measurement (loss_meas_bool)
             * Contact Force Measurement (loss_meas_force)
             * Contact Normal Measurement (loss_meas_normal)

        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
        return None

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
                {k: v[..., 1:, :] for k, v in meas_contact_forces.items()},
                {k: v[..., 1:, :] for k, v in meas_contact_normals.items()},
                forward_args[2],  # estimated contact forces
                forward_args[3],  # estimated contact normals
                forward_args[4],  # estimated phi(t)
            )

        # VIMP needs timestamps, plant trajectory, and control
        return self._loss_vimp(
            {k: v[..., 1:, :] for k, v in meas_contact_forces.items()},
            {k: v[..., 1:, :] for k, v in meas_contact_normals.items()},
            timestamps,
            forward_args[0],  # plant states
            forward_args[1],  # plant control
        )

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
    def get_learned_geometry(self) -> Shape:
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
        return PydrakeToCollisionGeometryFactory.reverse_convert(body_geometry)

    @torch.no_grad
    def get_body_geometry(self, body_name: str) -> Shape:
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
        return self._learned_trajectory.get_current_pose_traj()

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
