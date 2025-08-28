#!/usr/bin/env python3

"""Utility functions for defining and optimizing actions

The main contents of this file are as follows:

    * Dataclass to define Workspace
    * Dataclass defining an action (plus converting into a raw trajectory)
"""

from dataclasses import dataclass, field
import random
from typing import Optional, Callable

import gin
import numpy as np
from scipy.spatial.transform import Rotation
from tensordict import TensorDictBase, TensorDict
import torch
from torch import Tensor


## Action / Workspace Parameters
@gin.configurable
@dataclass
class ActionWorkspaceParams:
    """Class to specify workspace setup and discrete action space"""

    # pylint: disable=too-many-instance-attributes
    workspace_xy_center: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    """2D Center of the workspace"""
    workspace_radius: float = 0.15  # m
    r"""Radius of workspace"""
    robot_radius: float = 0.01575  # m
    r"""Radius of robot spheres"""
    fixed_240_w: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0])
    )  # m
    r"""Where to keep the trifinger's unused 240deg arm"""
    approach_radius: float = 0.1  # m
    r"""How far way to start finger from center of object"""
    finger_0_vec: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    finger_120_vec: np.ndarray = field(
        default_factory=lambda: np.array([-1.0, 0.0, 0.0])
    )
    r"""Preferred approach axis for each finger"""

    # Switches
    ground_buffer: float = True
    r"""Truncate bottom of workspace at multiple of robot radius"""
    robot_buffer: bool = False
    r"""Target separate planes for both robot fingers to guarantee no contact"""

    def get_finger_vec(self, finger_idx: int) -> np.ndarray:
        """Get finger vector from finger index"""
        assert finger_idx < 2, f"Requested nonexistent finger {finger_idx}"
        if finger_idx == 0:
            return self.finger_0_vec
        return self.finger_120_vec

    def get_reset_knot(self) -> np.ndarray:
        """Get safe positions to reset trifinger"""
        ret = np.zeros(18)
        ret[:3] = self.finger_0_vec * self.workspace_radius
        ret[3:6] = self.finger_120_vec * self.workspace_radius
        ret[6:9] = self.fixed_240_w
        if self.ground_buffer:
            ret[2] = np.clip(ret[2], a_min=self.robot_radius, a_max=None)
            ret[5] = np.clip(ret[5], a_min=self.robot_radius, a_max=None)
        return ret

    def __post_init__(self):
        """Method to check validity of parameters."""
        assert self.workspace_radius > 0.0
        assert 0.0 < self.robot_radius < self.workspace_radius
        assert len(self.workspace_xy_center) == 2
        assert len(self.fixed_240_w) == 3
        assert np.isclose(np.linalg.norm(self.finger_0_vec), 1.0)
        assert np.isclose(np.linalg.norm(self.finger_120_vec), 1.0)


N_ACTION_PARAMS = 4


@gin.configurable
@dataclass
class Action:
    """Action specification.

    Each action is defined by R^4
    For each finger: a right-ascension (i.e. rotation of primary axis about Z)
        and a declination (a rotation towards or away from +Z)

    """

    # pylint: disable=too-many-instance-attributes
    finger_0_ra: float = 0.0
    finger_0_dec: float = 0.0
    finger_120_ra: float = 0.0
    finger_120_dec: float = 0.0

    def get_ra_dec(self, finger_idx: int) -> np.ndarray:
        """Get RA and Dec as 2d array for a given finger idx"""
        assert finger_idx < 2, f"Requested nonexistent finger {finger_idx}"
        if finger_idx == 0:
            return np.array([self.finger_0_ra, self.finger_0_dec])
        return np.array([self.finger_120_ra, self.finger_120_dec])

    def get_params(self) -> np.ndarray:
        """Get params as a numpy array"""
        return np.array(
            [
                self.finger_0_ra,
                self.finger_0_dec,
                self.finger_120_ra,
                self.finger_120_dec,
            ]
        )

    # Order is arbitrary
    def __lt__(self, _):
        return bool(random.getrandbits(1))

    def __str__(self):
        return f"F0: ({self.finger_0_ra:.2f}, {self.finger_0_dec:.2f};\
            F120: ({self.finger_120_ra:.2f}, {self.finger_120_dec:.2f}"

    def __post_init__(self):
        """Method to check validity of parameters."""
        self.finger_0_ra = np.clip(self.finger_0_ra, -np.pi / 2.0, np.pi / 2.0)
        self.finger_0_dec = np.clip(self.finger_0_dec, -np.pi / 18.0, np.pi / 2.0)
        self.finger_120_ra = np.clip(self.finger_120_ra, -np.pi / 2.0, np.pi / 2.0)
        self.finger_120_dec = np.clip(self.finger_120_dec, -np.pi / 18.0, np.pi / 2.0)


@gin.configurable
class ActionCEM:
    """Class to handle CEM sampling of Actions"""

    # pylint: disable=too-many-instance-attributes, too-many-arguments, too-many-positional-arguments

    _rng: np.random.Generator

    # Current Distribution
    _init_mean: np.ndarray
    _init_cov: np.ndarray

    # CEM Parameters
    _n_samples: int
    _n_dist: int
    _n_iter: int

    def __init__(
        self,
        init_mean: Optional[list[float]] = None,
        init_var: Optional[float] = None,
        n_samples: int = 50,
        n_dist: int = 10,
        n_iter: int = 3,
    ) -> None:
        self._init_mean = (
            np.zeros(N_ACTION_PARAMS) if init_mean is None else np.array(init_mean)
        )
        assert self._init_mean.shape == (N_ACTION_PARAMS,)

        self._init_cov = (
            np.eye(N_ACTION_PARAMS) * (np.pi / 2.0)
            if init_var is None
            else np.eye(N_ACTION_PARAMS) * init_var
        )
        assert self._init_cov.shape == (N_ACTION_PARAMS, N_ACTION_PARAMS)

        self._mean = np.copy(self._init_mean)
        self._cov = np.copy(self._init_cov)

        self._n_samples = n_samples
        assert self._n_samples > 0

        self._n_dist = n_dist
        assert 0 < self._n_dist < self._n_samples

        self._n_iter = n_iter
        assert 0 < self._n_iter

        self._rng = np.random.default_rng()

    def best_action(
        self,
        score_fn: Callable[[list[Action]], list[float]],
        final_iter_argmax: bool = True,
        vis_fn: Optional[Callable[[list[Action]], None]] = None,
    ) -> Action:
        """Run CEM to determine the best action to take"""

        mean = np.copy(self._init_mean)
        cov = np.copy(self._init_cov)

        for _ in range(self._n_iter):
            ## Sample Batch of Actions
            batch_actions = [
                Action(*param.tolist())
                for param in self._rng.multivariate_normal(
                    mean, cov, size=self._n_samples
                )
            ]
            assert len(batch_actions) == self._n_samples

            ## Visualize actions
            if vis_fn is not None:
                vis_fn(batch_actions)

            ## Score each action and sort
            _, sorted_actions = zip(
                *sorted(zip(score_fn(batch_actions), batch_actions), reverse=True)
            )

            ## Take best N actions and create new mean and covariance
            best_actions = sorted_actions[: self._n_dist]
            best_params = np.stack([act.get_params() for act in best_actions])
            assert best_params.shape == (self._n_dist, N_ACTION_PARAMS)
            mean = np.mean(best_params, axis=0)
            assert mean.shape == (N_ACTION_PARAMS,)
            cov = np.cov(
                best_params, rowvar=False
            )  # Each col is a variable, each row is a sample
            assert cov.shape == (N_ACTION_PARAMS, N_ACTION_PARAMS)

        if final_iter_argmax:
            # Take the best action from the last batch
            return sorted_actions[0]

        # Take the mean
        return Action(*mean.tolist())


@gin.configurable
def action_to_knots(
    params: ActionWorkspaceParams,
    actions: list[Action],
    object_pose_estimate: np.ndarray,
    include_finger_240: bool = True,
):
    """
    Convert Actions into knot points

    Returns:
        np.ndarray (len(actions), n_knots, n_fingers * 6 in drake order [q then v])
        Likely (len(actions), 2, 18) or (len(actions), 2, 12)
    """

    # pylint: disable=too-many-locals

    # Input Validation
    assert params is not None
    assert len(actions) > 0
    assert object_pose_estimate.shape == (7,)
    object_position = object_pose_estimate[4:]

    # Velocity == 0
    ret2 = np.zeros((len(actions), 2, 12))

    # Get start knot point from action definition
    for finger_idx in range(2):
        ra_axis = np.array([0.0, 0.0, 1.0])
        pref_axis = params.get_finger_vec(finger_idx)
        assert np.isclose(np.linalg.norm(pref_axis), 1.0)
        dec_axis = np.cross(pref_axis, ra_axis).reshape(1, 3)
        assert np.linalg.norm(dec_axis) > 0.0, "Preferred finger axis parallel to +Z"
        finger_ras_decs = np.stack([act.get_ra_dec(finger_idx) for act in actions])
        assert finger_ras_decs.shape == (len(actions), 2)
        dec_rotvecs = Rotation.from_rotvec(
            dec_axis / np.linalg.norm(dec_axis) * finger_ras_decs[:, 1:]
        )
        ra_rotvecs = Rotation.from_rotvec(
            ra_axis.reshape(1, 3) * finger_ras_decs[:, :1]
        )

        # Rotate preferred axis and scale to fixed distance
        start_poses = params.approach_radius * ra_rotvecs.apply(
            dec_rotvecs.apply(pref_axis)
        )
        assert start_poses.shape == (len(actions), 3)
        ret2[:, 0, finger_idx * 3 : (finger_idx + 1) * 3] = start_poses

    # End knot point and all velocities == 0

    # Translate start point to object position
    ret2[:, :, :6] += np.tile(object_position, 2)

    # Clip to workspace edge
    # TODO: use ray-sphere intersection
    for finger_idx in range(2):
        start_poses = ret2[:, 0, finger_idx * 3 : (finger_idx + 1) * 3]
        end_poses = ret2[:, 1, finger_idx * 3 : (finger_idx + 1) * 3]
        norm_ratio = np.clip(
            np.linalg.norm(start_poses, axis=-1),
            a_min=None,
            a_max=params.workspace_radius,
        ) / np.linalg.norm(start_poses, axis=-1)
        norm_ratio[np.isnan(norm_ratio)] = 0.0
        start_poses = start_poses * norm_ratio.reshape(-1, 1)
        # Clip to ground
        if params.ground_buffer:
            start_poses[:, 2] = np.clip(
                start_poses[:, 2],
                a_min=params.robot_radius,
                a_max=None,
            )
            end_poses[:, 2] = np.clip(
                end_poses[:, 2],
                a_min=params.robot_radius,
                a_max=None,
            )

        ret2[:, 0, finger_idx * 3 : (finger_idx + 1) * 3] = start_poses
        ret2[:, 1, finger_idx * 3 : (finger_idx + 1) * 3] = end_poses

    # Add displacement along preferred axis to guarantee no contact (if preferred axes are opposing)
    if params.robot_buffer:
        pref_0 = params.get_finger_vec(0)
        pref_120 = params.get_finger_vec(1)

        # Start Separation
        start_0 = np.copy(ret2[:, 0, :3])
        start_120 = np.copy(ret2[:, 0, 3:6])
        start_dists = np.linalg.norm(start_0 - start_120, axis=-1)
        start_penetration = np.clip(
            2.0 * params.robot_radius - start_dists, a_min=0.0, a_max=None
        )
        new_start_0 = start_0 + (
            start_penetration[:, np.newaxis] * 0.5 * pref_0[np.newaxis, :]
        )
        new_start_120 = start_120 + (
            start_penetration[:, np.newaxis] * 0.5 * pref_120[np.newaxis, :]
        )
        ret2[:, 0, :3] = new_start_0
        ret2[:, 0, 3:6] = new_start_120

        # End Separation
        end_0 = np.copy(ret2[:, 1, :3])
        end_120 = np.copy(ret2[:, 1, 3:6])
        # First separate by start penetration
        new_end_0 = end_0 + (
            start_penetration[:, np.newaxis] * 0.5 * pref_0[np.newaxis, :]
        )
        new_end_120 = end_120 + (
            start_penetration[:, np.newaxis] * 0.5 * pref_120[np.newaxis, :]
        )
        end_dists = np.linalg.norm(new_end_0 - new_end_120, axis=-1)
        end_penetration = np.clip(
            2.0 * params.robot_radius - end_dists, a_min=0.0, a_max=None
        )
        # Separate along start-end diretion
        dir_0 = (new_start_0 - new_end_0) / np.linalg.norm(
            new_start_0 - new_end_0, axis=-1, keepdims=True
        )
        dir_120 = (new_start_120 - new_end_120) / np.linalg.norm(
            new_start_120 - new_end_120, axis=-1, keepdims=True
        )
        new_new_end_0 = new_end_0 + (end_penetration[:, np.newaxis] * 0.5 * dir_0)
        new_new_end_120 = new_end_120 + (end_penetration[:, np.newaxis] * 0.5 * dir_120)

        ret2[:, 1, :3] = new_new_end_0
        ret2[:, 1, 3:6] = new_new_end_120

    # Add fixed 240 position
    if include_finger_240:
        ret = np.zeros((len(actions), 2, 18))
        ret[:, :, :6] = ret2[:, :, :6]
        ret[:, :, 6:9] = np.tile(params.fixed_240_w, ret.shape[:-1] + (1,))
    else:
        ret = ret2

    return ret


@gin.configurable(denylist=["data"])
def interpolate_sampled_action(
    data: Tensor, fingertip_body_names: list[str], traj_len_s=2.0, traj_n_steps=61
) -> TensorDictBase:
    """
    Interpolates a start/end action using a cubic spline.

    Params:
        data: outputs of action_to_knots size (batch, 2, 18)

    Returns:
        TensorDict input of extract_robot_trajectory, batch_size=(batch, traj_n_steps), keys = fingertip_body_names
        Timestamps = Tensor size (traj_n_steps,)
    """
    # pylint: disable=too-many-locals

    assert len(data.size()) >= 2
    batch_dims = data.size()[:-2]
    assert data.size() == batch_dims + (2, 18)
    ret = TensorDict({}, batch_size=batch_dims + (traj_n_steps,))
    ret_timestamps = torch.linspace(0.0, traj_len_s, traj_n_steps)  # (traj_n_steps,)
    rel_timestamps = (
        (ret_timestamps - ret_timestamps[0]) / (ret_timestamps[-1] - ret_timestamps[0])
    ).unsqueeze(
        0
    )  # (1, traj_n_steps)
    samples = data[..., :, :9]  # (batch, 2, 9)
    samples_dot = data[..., :, 9:]  # (batch, 2, 9)
    spline_a = samples[..., 0, :].unsqueeze(-1)  # (batch, 9, 1)
    spline_b = samples_dot[..., 0, :].unsqueeze(-1)  # (batch, 9, 1)
    spline_c = (
        3.0 * (samples[..., 1, :] - samples[..., 0, :])
        - 2.0 * samples_dot[..., 0, :]
        - samples_dot[..., 1, :]
    ).unsqueeze(
        -1
    )  # (batch, 9, 1)
    spline_d = (
        2.0 * (samples[..., 0, :] - samples[..., 1, :])
        + samples_dot[..., 0, :]
        + samples_dot[..., 1, :]
    ).unsqueeze(
        -1
    )  # (batch, 9, 1)

    # (batch, traj_n_steps, 9)
    data_lerp = (
        spline_a @ torch.pow(rel_timestamps, 0.0)
        + spline_b @ torch.pow(rel_timestamps, 1.0)
        + spline_c @ torch.pow(rel_timestamps, 2.0)
        + spline_d @ torch.pow(rel_timestamps, 3.0)
    )
    data_lerp = torch.transpose(data_lerp, -1, -2)
    data_lerp_dot = (
        spline_b @ torch.pow(rel_timestamps, 0.0)
        + 2.0 * spline_c @ torch.pow(rel_timestamps, 1.0)
        + 3.0 * spline_d @ torch.pow(rel_timestamps, 2.0)
    )
    data_lerp_dot = torch.transpose(data_lerp_dot, -1, -2)

    for fingertip in fingertip_body_names:
        ret[fingertip, "position"] = torch.zeros(batch_dims + (traj_n_steps, 3))
        ret[fingertip, "velocity"] = torch.zeros(batch_dims + (traj_n_steps, 3))
    for finger_idx, fingertip in enumerate(fingertip_body_names):
        pos_idx = 3 * finger_idx
        ret[fingertip, "position"][..., :] = data_lerp[..., pos_idx : pos_idx + 3]
        ret[fingertip, "velocity"][..., :] = data_lerp_dot[..., pos_idx : pos_idx + 3]

    return ret, ret_timestamps
