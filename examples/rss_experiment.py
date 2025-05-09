#!/usr/bin/env python3
"""
Entry Point for Exploration Experiment

TODOs:
* Execute a trajectory on robot / sim via LCM and record data
* Given data, train an object estimate.
* * Run forward_dynamics to create initial trajectory guess
* * Pytorch AutoDiff for SGD
* Given a learned system
* * Generate the Fisher Information
* * Sample robot trajectory
* Given a learned system + ground truth object data:
* * calculate accuracy metrics (Chamfer distance + co-located chamfer distance + position error)
"""

# pylint: disable=invalid-name,too-many-statements,too-many-locals

from enum import Enum
import os
import pdb
import random
import signal
import sys
import time
from typing import cast, Any, Dict, List, Optional, Tuple, Type

import gin
import gin.torch.external_configurables
import git
import lcm
import numpy as np
from pydrake.all import StartMeshcat, Rgba, Shape
from pydrake.geometry import HalfSpace as DrakeHalfSpace  # type: ignore
from scipy.spatial.transform import Rotation as R
from tensordict import TensorDictBase, TensorDict
import torch
from torch.distributions.normal import Normal
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch import Tensor

# tensor_utils required for TensorDict's collate_fn
# pylint: disable-next=unused-import
from dair_pll import tensor_utils
from dair_pll import file_utils
from dair_pll.drake_system import DrakeSystem
from dair_pll.dataset_management import TrajectorySet
from dair_pll.gui_utils import PLLMeshcatVisualizer
from dair_pll.multibody_learnable_system import MultibodyLearnableSystemWithTrajectory
from dair_pll.lcmtypes.dairlib import (
    lcmt_fingertips_position,
    lcmt_object_state,
    lcmt_densetact_measurement_data,
    lcmt_fingertips_target_kinematics,
)
from dair_pll.tensor_utils import pbmm
from dair_pll.hack_utils import finger_idx_from_body_name

# Repository directory (default for file operations)
REPO_DIR = os.path.normpath(
    git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")
)
DEFAULT_CONFIG = "rss_experiment.gin"

## Training Functions
@gin.configurable
def get_loss_args(
    x_past: Tensor,
    x_future: Tensor,
    system: DrakeSystem,
    impulses: Optional[Tensor] = None,
    object_body_name: str = "cube",
    ground_std: float = 1e-8,
) -> Dict[str, Any]:
    """Convert dataloader trajectory slices into arguments for contactnets loss"""

    # Get last time of past and first of future
    # Remove extraneous dimensions
    past = x_past[..., -1]
    plus = x_future[..., 0]

    # Construct State
    x_past = system.construct_state_tensor(past)
    x_plus = system.construct_state_tensor(plus)

    # Add Ground Noise
    sampler_ground = Normal(loc = 0., scale = ground_std)
    state_names = system.multibody_terms.plant_diagram.plant.GetStateNames()
    z_mask = torch.tensor([s.endswith("z_x") or s.endswith("_z") for s in state_names])
    sample = sampler_ground.sample(x_past[..., z_mask].size())
    x_past[..., z_mask] += sample
    x_plus[..., z_mask] += sample

    # Actuation
    n_control = system.plant_diagram.plant.num_actuated_dofs()
    control = torch.zeros(past.batch_size + (n_control,))
    if "net_actuation" in past.keys():
        control = past["net_actuation"]
        if len(control.shape) == 1:
            control = control.unsqueeze(-1)

    # Construct measured contact forces on obj_b from obj_a
    # Defined as Dict: {(str(obj_a_name), str(obj_b_name)) -> R^3 force on obj_b in World Frame}
    # TODO: specify incoming data reference frame, default World
    contact_forces = {}
    if "contact_forces" in past.keys():
        for key in past["contact_forces"].keys():
            contact_forces[(object_body_name, key)] = past["contact_forces"][key]
    contact_normals = {}
    if "contact_normals" in past.keys():
        for key in past["contact_normals"].keys():
            contact_normals[(object_body_name, key)] = past["contact_normals"][key]

    ret = {
        "x": x_past,
        "u": control,
        "x_plus": x_plus,
        "contact_forces": contact_forces,
        "contact_normals": contact_normals,
    }
    if impulses is not None:
        ret["impulses"] = impulses

    return ret

def train_epoch(
    data: DataLoader,
    system: MultibodyLearnableSystemWithTrajectory,
    optimizer: Optional[Optimizer] = None,
) -> Tensor:
    """Train learned model for a single epoch.  Takes gradient steps in the
    learned parameters if ``optimizer`` is provided.

    Args:
        data: Training dataset.
        system: System to be trained.
        optimizer: Optimizer which trains system.

    Returns:
        Scalar average training loss observed during epoch.
    """
    losses = []
    loss_elements = {}
    for xy_i in data:
        x_past: Tensor = xy_i[0]
        x_plus: Tensor = xy_i[1]

        if optimizer is not None:
            optimizer.zero_grad()
        # pylint: disable=E1120
        # Expect gin to handle missing arguments
        ### Profiling
        #import cProfile, pstats, io
        #from pstats import SortKey
        #pr = cProfile.Profile()
        #pr.enable()
        loss = system.contactnets_loss(**get_loss_args(x_past, x_plus, system)).mean()
        losses.append(loss.clone().detach())

        for key, val in system.loss_cache.items():
            if key not in loss_elements:
                loss_elements[key] = []
            loss_elements[key].append(val)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        ### Profiling
        #s = io.StringIO()
        #sortby = SortKey.CUMULATIVE
        #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #ps.print_stats()
        #print(s.getvalue())
        #breakpoint()

    # Compute Epoch Average
    avg_loss = cast(Tensor, sum(losses) / len(losses))
    loss_elements_ret = {}
    for key, val in loss_elements.items():
        loss_elements_ret[key] = cast(Tensor, sum(val) / len(val))
    return avg_loss, loss_elements_ret


## Execute Robot Trajectory
@gin.configurable
class TrifingerLCMService:
    """
    Command robot and collect data over LCM
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        lcm_channels: Dict[str, str],
        fingertip_body_names: List[str],
        traj_time_len=2.0,
    ):
        self._lcm_channels = lcm_channels
        self._traj_time_len = traj_time_len
        self._fingertip_body_names = fingertip_body_names

        self._force_raw_data = []
        self._fingertip_pose_raw_data = []
        self._object_raw_data = []

        # Init LCM Subscriptions
        self._lcm = lcm.LCM()
        self._lcm_subs = {}
        self._lcm_subs["fingertips_position"] = self._lcm.subscribe(
            lcm_channels["fingertips_position"], self.sub_handler
        )
        self._lcm_subs["densetact"] = self._lcm.subscribe(
            lcm_channels["densetact"], self.sub_handler
        )
        self._lcm_subs["object_state"] = self._lcm.subscribe(
            lcm_channels["object_state"], self.sub_handler
        )
        for sub in self._lcm_subs.values():
            sub.set_queue_capacity(
                1
            )  # to discard everything outside of the handle window

    def sub_handler(self, channel: str, data: Any):
        """
        Write LCM incoming messages to cache
        """
        if channel == self._lcm_channels["fingertips_position"]:
            self._fingertip_pose_raw_data.append(lcmt_fingertips_position.decode(data))
        if channel == self._lcm_channels["densetact"]:
            self._force_raw_data.append(lcmt_densetact_measurement_data.decode(data))
        if channel == self._lcm_channels["object_state"]:
            self._object_raw_data.append(lcmt_object_state.decode(data))

    def execute_trajectory(
        self,
        target_state: np.ndarray,
        pos_is_absolute: bool = True,
        no_data: bool = False,
    ) -> TensorDictBase:
        """
        Direct the robot to go to target_state.
        Record all incoming data over the next traj_time_len seconds.

        NOTE: assumes that target_state is in order 
            (finger_0q, finger_120q, finger_240q, finger_0v, finger_120v, finger_240v)
        """
        # pylint: disable=too-many-locals
        command = lcmt_fingertips_target_kinematics()
        assert target_state.shape == (len(command.targetPos) + len(command.targetVel),)
        command.utime = int(time.time() * 1e6)
        command.isAbsoluteTargetPos = pos_is_absolute
        command.targetPos[:] = target_state[: len(command.targetPos)]
        command.targetVel[:] = target_state[len(command.targetPos) :]

        # Start with clear data
        self._force_raw_data.clear()
        self._fingertip_pose_raw_data.clear()
        self._object_raw_data.clear()

        print(f"Sending Command at: {time.time()}")
        self._lcm.publish(self._lcm_channels["fingertips_target"], command.encode())
        end_time = time.time() + self._traj_time_len
        while time.time() < end_time:
            self._lcm.handle_timeout(int((end_time - time.time()) * 1e3))
        print(f"Finished at: {time.time()}")
        print(
            f"Collected {len(self._fingertip_pose_raw_data)}" +
            f" / {len(self._force_raw_data)} / {len(self._object_raw_data)} samples."
        )

        # Return empty if not any force data
        ret = TensorDict({}, batch_size=len(self._force_raw_data))
        if no_data or len(self._force_raw_data) < 1:
            return ret

        assert self._force_raw_data[0].numSensors == len(self._fingertip_body_names)
        assert len(self._fingertip_pose_raw_data) >= len(self._force_raw_data)
        def is_sorted(a: np.ndarray) -> bool:
            return np.all(a[:-1] <= a[1:])
        densetact_time_s = np.array(
            [
                float(measurement.sensorData[0].timestamp) / 1e6
                for measurement in self._force_raw_data
            ]
        ).flatten()
        assert is_sorted(densetact_time_s)
        fingerpos_time_s = np.array(
            [
                float(measurement.utime) / 1e6
                for measurement in self._fingertip_pose_raw_data
            ]
        ).flatten()
        assert is_sorted(fingerpos_time_s)
        object_time_s = np.array(
            [float(measurement.utime) / 1e6 for measurement in self._object_raw_data]
        ).flatten()
        assert is_sorted(object_time_s)

        # Interp fingertip data
        fingertip_pos_W = {}
        fingertip_vel_W = {}
        fingertip_force_C = {}
        fingertip_force_W = {}
        fingertip_normal_W = {}
        for body_idx, body_name in enumerate(self._fingertip_body_names):
            # Position Interpolation
            body_pos = np.array(
                [
                    measurement.curPos[3 * body_idx : 3 * body_idx + 3]
                    for measurement in self._fingertip_pose_raw_data
                ]
            )
            assert body_pos.shape == (len(fingerpos_time_s), 3)
            body_pos_interp = np.vstack(
                [
                    np.interp(densetact_time_s, fingerpos_time_s, body_pos[:, idx])
                    for idx in range(3)
                ]
            ).T
            assert body_pos_interp.shape == (len(densetact_time_s), 3)
            fingertip_pos_W[body_name] = body_pos_interp

            # Velocity Interpolation
            body_vel = np.array(
                [
                    measurement.curVel[3 * body_idx : 3 * body_idx + 3]
                    for measurement in self._fingertip_pose_raw_data
                ]
            )
            assert body_vel.shape == (len(fingerpos_time_s), 3)
            body_vel_interp = np.vstack(
                [
                    np.interp(densetact_time_s, fingerpos_time_s, body_vel[:, idx])
                    for idx in range(3)
                ]
            ).T
            assert body_vel_interp.shape == (len(densetact_time_s), 3)
            fingertip_vel_W[body_name] = body_vel_interp

            # Quat Interpolation
            body_quat = np.array(
                [
                    measurement.curQuat[4 * body_idx : 4 * body_idx + 4]
                    for measurement in self._fingertip_pose_raw_data
                ]
            )
            assert body_quat.shape == (len(fingerpos_time_s), 4)
            body_quat_interp = np.vstack(
                [
                    np.interp(densetact_time_s, fingerpos_time_s, body_quat[:, idx])
                    for idx in range(4)
                ]
            ).T
            assert body_quat_interp.shape == (len(densetact_time_s), 4)
            body_R_BW = R.from_quat(body_quat_interp, scalar_first=True)

            # Record normal and force in world frame
            body_R_CB = R.from_matrix(
                np.stack(
                    [
                        np.array(measurement.sensorData[body_idx].contactFrame)[:3, :3]
                        for measurement in self._force_raw_data
                    ]
                )
            )
            normal_C = np.broadcast_to(
                np.array([0.0, 0.0, 1.0]), (len(densetact_time_s), 3)
            )
            body_R_CW = body_R_BW.inv() * body_R_CB
            fingertip_normal_W[body_name] = body_R_CW.apply(normal_C)
            # Zero out no contact normal
            finger_in_contact = np.array([
                measurement.sensorData[body_idx].inContact
                        for measurement in self._force_raw_data
                ])
            fingertip_normal_W[body_name][~finger_in_contact] = 0.
            force_C = np.array(
                [
                    (
                        list(measurement.sensorData[body_idx].scaledFriction)
                        + [measurement.sensorData[body_idx].scaledNormal]
                    )
                    for measurement in self._force_raw_data
                ]
            )
            assert force_C.shape == (len(densetact_time_s), 3)
            fingertip_force_W[body_name] = body_R_CW.apply(force_C)
            fingertip_force_C[body_name] = force_C

        ret["time"] = torch.from_numpy(densetact_time_s)
        for body_name in self._fingertip_body_names:
            ret[body_name, "position"] = torch.from_numpy(
                fingertip_pos_W[body_name]
            ).clone()
            ret[body_name, "velocity"] = torch.from_numpy(
                fingertip_vel_W[body_name]
            ).clone()
            ret[body_name, "contact_force_C"] = torch.from_numpy(
                fingertip_force_C[body_name]
            ).clone()
            ret[body_name, "contact_force_W"] = torch.from_numpy(
                fingertip_force_W[body_name]
            ).clone()
            ret[body_name, "contact_normal_W"] = torch.from_numpy(
                fingertip_normal_W[body_name]
            ).clone()

        # Interp ground-truth object data
        if len(self._object_raw_data) > 0:
            # Position Interpolation
            num_positions = self._object_raw_data[0].num_positions
            object_pos = np.array(
                [measurement.position[:] for measurement in self._object_raw_data]
            )
            assert object_pos.shape == (len(object_time_s), num_positions)
            object_pos_interp = np.vstack(
                [
                    np.interp(densetact_time_s, object_time_s, object_pos[:, idx])
                    for idx in range(num_positions)
                ]
            ).T
            assert object_pos_interp.shape == (len(densetact_time_s), num_positions)
            ret[self._object_raw_data[0].object_name, "position"] = torch.from_numpy(
                object_pos_interp
            ).clone()

            # Velocity Interpolation
            num_velocities = self._object_raw_data[0].num_velocities
            object_vel = np.array(
                [measurement.velocity[:] for measurement in self._object_raw_data]
            )
            assert object_vel.shape == (len(object_time_s), num_velocities)
            object_vel_interp = np.vstack(
                [
                    np.interp(densetact_time_s, object_time_s, object_vel[:, idx])
                    for idx in range(num_velocities)
                ]
            ).T
            assert object_vel_interp.shape == (len(densetact_time_s), num_velocities)
            ret[self._object_raw_data[0].object_name, "velocity"] = torch.from_numpy(
                object_vel_interp
            ).clone()

        # Return
        return ret


class ActionLibrary(Enum):
    NONE = 0
    XPINCH = 1
    YPINCH = 2
    ZPINCH = 3
    XSINGLE = 4
    YSINGLE = 5
    ZSINGLE = 6
    CORNERSINGLE = 7
    EDGESINGLE = 8

@gin.configurable
def sample_action(
    workspace_xy_center: Tuple[float, float],
    workspace_z_rot: float,
    workspace_radius: float,
    sphere_radius: float,
    fixed_240_W: List[float],
    library: ActionLibrary = ActionLibrary.NONE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a straight line action
    Params:
    workspace_xy_center: offset the workspace from world origin
    workspaxe_z_rot: world rotation so finger_0 is towards +X axis
    workspace_radius: start will be along edge of radius above ground
    sphere_radius: radius of robot fingertip
    fixed_240: 3d position
    library: int for fixed set of action
    """

    assert workspace_radius > 0.0
    assert 0.0 < sphere_radius < workspace_radius
    assert len(fixed_240_W) == 3
    fixed_240_traj = np.array(fixed_240_W)
    rng = np.random.default_rng()

    # Start in workspace frame
    def sample_finger(flip_x: bool = False, library = library):
        flip_factor = -1.0 if flip_x else 1.0
        start_polar = rng.uniform(0.0, np.pi / 2.0)
        start_azimuth = rng.uniform(-np.pi / 2.0, np.pi / 2.0)
        if library in (ActionLibrary.XPINCH, ActionLibrary.XSINGLE):
            start_polar = np.pi / 2.0
            start_azimuth = 0.
        elif library in (ActionLibrary.ZPINCH, ActionLibrary.ZSINGLE):
            start_polar = 0.
            start_azimuth = 0.
        elif library in (ActionLibrary.YPINCH, ActionLibrary.YSINGLE):
            start_polar = np.pi / 2.0
            start_azimuth = flip_factor * (np.pi / 2.0)
        elif library in (ActionLibrary.CORNERSINGLE,):
            start_polar = np.pi / 5.0
            start_azimuth = np.pi / 2.5
        elif library in (ActionLibrary.EDGESINGLE,):
            start_polar = np.pi / 7.0
            start_azimuth = 0.
        if flip_x and (library in (ActionLibrary.XSINGLE, ActionLibrary.YSINGLE, ActionLibrary.CORNERSINGLE, ActionLibrary.EDGESINGLE)):
            start_polar = 0.
            start_azimuth = 0.
        if flip_x and (library in (ActionLibrary.ZSINGLE,)):
            start_polar = np.pi / 2.0
            start_azimuth = -np.pi / 2.0
        start_S = (workspace_radius - sphere_radius) * np.array(
            [
                (np.sin(start_polar) * np.cos(start_azimuth)),
                np.sin(start_polar) * np.sin(start_azimuth),
                np.cos(start_polar),
            ]
        )
        #start_S[0] += sphere_radius
        start_S[2] += sphere_radius
        start_S[0] *= flip_factor
        max_radius = workspace_radius - sphere_radius
        end_radius = rng.uniform(0.0, max_radius)
        end_angle = rng.uniform(0.0, np.pi)
        if not (library is ActionLibrary.NONE):
            end_radius = 0.
            end_angle = 0.
        if flip_x and (library in (ActionLibrary.XSINGLE, ActionLibrary.YSINGLE, ActionLibrary.ZSINGLE, ActionLibrary.EDGESINGLE, ActionLibrary.CORNERSINGLE)):
            end_radius = max_radius
            end_angle = np.pi / 2.0
        if flip_x and (library in (ActionLibrary.ZSINGLE,)):
            end_radius = max_radius
            end_angle = np.pi
        end_S = np.array(
            [
                flip_factor * 0.,#sphere_radius,
                end_radius * np.cos(end_angle),
                sphere_radius + end_radius * np.sin(end_angle),
            ]
        )
        return (start_S, end_S)

    finger_0_traj = sample_finger(False)
    finger_120_traj = sample_finger(True)

    ret = (np.zeros(18), np.zeros(18))
    z_rot = np.array(
        [
            [np.cos(workspace_z_rot), -np.sin(workspace_z_rot), 0.0],
            [np.sin(workspace_z_rot), np.cos(workspace_z_rot), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    xy_trans = np.array([workspace_xy_center[0], workspace_xy_center[1], 0.0])
    for idx in [0, 1]:
        ret[idx][:3] = z_rot @ finger_0_traj[idx].T + xy_trans
        ret[idx][3:6] = z_rot @ finger_120_traj[idx].T + xy_trans
        ret[idx][6:9] = fixed_240_traj[:]
    return ret


@gin.configurable(denylist=["data"])
def interpolate_sampled_action(
    data: Tensor, fingertip_body_names: List[str], traj_len_s=2.0, traj_n_steps=61
) -> TensorDictBase:
    """
    Interpolates a start/end action using a cubic spline.

    Params:
        data: outputs of sample_action size (batch, 2, 18)

    Returns:
        TensorDict input of extract_robot_trajectory, batch_size=(batch, traj_n_steps), keys = fingertip_body_names
        Timestamps = Tensor size (traj_n_steps,)
    """
    assert len(data.size()) >= 2
    batch_dims = data.size()[:-2]
    assert data.size() == batch_dims + (2, 18)
    ret = TensorDict({}, batch_size=batch_dims + (traj_n_steps,))
    ret_timestamps = torch.linspace(0.0, traj_len_s, traj_n_steps) # (traj_n_steps,)
    rel_timestamps = ((ret_timestamps - ret_timestamps[0]) / (ret_timestamps[-1] - ret_timestamps[0])).unsqueeze(0) # (1, traj_n_steps)
    samples = data[..., :, :9] # (batch, 2, 9)
    samples_dot = data[..., :, 9:] # (batch, 2, 9)
    spline_a = samples[..., 0, :].unsqueeze(-1) # (batch, 9, 1)
    spline_b = samples_dot[..., 0, :].unsqueeze(-1) # (batch, 9, 1)
    spline_c = (3.*(samples[..., 1, :]-samples[..., 0, :]) - 2.*samples_dot[..., 0, :] - samples_dot[..., 1, :]).unsqueeze(-1) # (batch, 9, 1)
    spline_d = (2.*(samples[..., 0, :]-samples[..., 1, :]) + samples_dot[..., 0, :] + samples_dot[..., 1, :]).unsqueeze(-1) # (batch, 9, 1)

    # (batch, traj_n_steps, 9)
    data_lerp = spline_a @ torch.pow(rel_timestamps, 0.) + spline_b @ torch.pow(rel_timestamps, 1.) + spline_c @ torch.pow(rel_timestamps, 2.) + spline_d @ torch.pow(rel_timestamps, 3.)
    data_lerp = torch.transpose(data_lerp, -1, -2)
    data_lerp_dot = spline_b @ torch.pow(rel_timestamps, 0.) + 2.*spline_c @ torch.pow(rel_timestamps, 1.) + 3.*spline_d @ torch.pow(rel_timestamps, 2.)
    data_lerp_dot = torch.transpose(data_lerp_dot, -1, -2)

    for fingertip in fingertip_body_names:
        ret[fingertip, "position"] = torch.zeros(batch_dims + (traj_n_steps, 3))
        ret[fingertip, "velocity"] = torch.zeros(batch_dims + (traj_n_steps, 3))
    for finger_idx, fingertip in enumerate(fingertip_body_names):
        pos_idx = 3 * finger_idx
        ret[fingertip, "position"][..., :] = data_lerp[..., pos_idx : pos_idx + 3]
        ret[fingertip, "velocity"][..., :] = data_lerp_dot[..., pos_idx : pos_idx + 3]

    return ret, ret_timestamps

@gin.configurable(denylist=["system", "data"])
def extract_robot_trajectory(
    system: MultibodyLearnableSystemWithTrajectory,
    data: TensorDictBase,
    robot_model_name: str,
    fingertip_body_names: List[str],
) -> Tensor:
    """
    Params:
        data: batched TensorDict from execute_trajectory()

    Returns:
        (batch, robot_space_nx)
    """

    assert len(data.size()) >= 1
    batch_dims = data.size()
    plant = system.plant_diagram.plant
    robot_space = system.get_model_space(robot_model_name)
    ret = torch.zeros(batch_dims + (robot_space.n_x,))

    for finger_name, state_idx in zip(
        fingertip_body_names,
        finger_idx_from_body_name(
            plant, plant.GetModelInstanceByName(robot_model_name), fingertip_body_names
        ),
    ):
        pos_idx = 3 * state_idx
        vel_idx = robot_space.n_x // 2 + pos_idx
        ret[..., pos_idx : pos_idx + 3] = data[finger_name]["position"]
        ret[..., vel_idx : vel_idx + 3] = data[finger_name]["velocity"]

    return ret


### Visualization
def get_true_geometry() -> Shape:
    """Get True Geometry from configured base system"""
    system = DrakeSystem()
    inspector = system.plant_diagram.scene_graph.model_inspector()
    all_geom_ids = inspector.GetAllGeometryIds()
    for geom_id in all_geom_ids:
        true_geom = inspector.GetShape(geom_id)
        if isinstance(true_geom, DrakeHalfSpace):
            continue
        return true_geom
    assert False, "Could not find true geometry"
    return None


### Signal Handling
signal_pressed = False
def signal_handler(sig, frame):
    """ Handle SIGINT"""
    global signal_pressed
    signal_pressed = True

## Main Function
@gin.configurable
def main(
    init_trifinger_state: List[float],
    safe_trifinger_height: float,
    robot_model_name: str,
    object_model_name: str = "cube",
    n_actions_optimized: int = 30,
    storage_folder_name: str = "storage_rss",
    run_name: str = "default_run",
    optimizer_cls: Type = torch.optim.SGD,
):
    """Main function for online learning loop"""
    global signal_pressed
    signal.signal(signal.SIGINT, signal_handler)
    #torch.autograd.set_detect_anomaly(True) ## NOTE: doesn't work with vmap
    torch.set_default_device("cuda")

    # Create run directory
    print("Active Tactile Exploration")
    storage_name = os.path.join(REPO_DIR, "results", storage_folder_name)
    print(f"Storing data and results at {file_utils.run_dir(storage_name, run_name)}")

    # Create learnable system
    print("Loading Learned System...")
    learned_system = MultibodyLearnableSystemWithTrajectory(
        output_urdfs_dir=file_utils.get_learned_urdf_dir(storage_name, run_name)
    )
    learned_summaries = [learned_system.summary({})]
    train_losses = []
    train_loss_data = []

    # Create Dataset
    data_trajectories = TrajectorySet()

    # GUI Visualization
    gui_vis = PLLMeshcatVisualizer(
        system = learned_system,
        data = data_trajectories,
        true_geom = get_true_geometry()
    )

    # Initialize LCM
    # Pylint doesn't know about gin
    # pylint: disable=no-value-for-parameter
    trifinger_lcm = TrifingerLCMService()
    print("Move to initial trifinger state")
    trifinger_lcm.execute_trajectory(np.array(init_trifinger_state), no_data=True)
    print("Sample Initial Random Action...")
    action_library = [ActionLibrary.XSINGLE, ActionLibrary.XPINCH, ActionLibrary.YSINGLE, ActionLibrary.YPINCH, ActionLibrary.ZSINGLE]#, ActionLibrary.EDGESINGLE, ActionLibrary.CORNERSINGLE]
    selected_action = sample_action(library=ActionLibrary.ZSINGLE)
    new_trajectory = None

    # Initialize Optimizer and Data config
    optimizer = optimizer_cls(learned_system.parameters())
    traj_dataloader = None
    obs_info_inv = None
    total_epochs = 0
    
    # Start Input Loop
    def print_help():
        print(
            "\nUsage:\n"
            "a - Action Selection\n"
            "e - Execute selected action + collect data\n"
            "o - Observed info\n"
            "s - Sample random action\n"
            "t - Train\n"
            "b - breakpoint()\n"
            "v - Visualize\n"
            "h - Print Help\n"
            "q - Quit\n"
        )

    print_help()
    command_char = " "
    while command_char != "q":
        command_char = input("Command $ ").split(" ")[0]

        if command_char == "h":
            print_help()

        elif command_char == "b":
            # pylint: disable-next=forgotten-debug-statement
            pdb.Pdb(nosigint=True).set_trace()

        elif command_char == "e":
            ## Execute selected action
            # Move to start state
            trifinger_lcm.execute_trajectory(selected_action[0], no_data=True)

            # Execute and collect data
            new_trajectory = trifinger_lcm.execute_trajectory(selected_action[1])

            if len(new_trajectory) < 1:
                print("WARNING: No data collected")
                continue

            # Move straight up
            safe_state = np.copy(selected_action[0])
            safe_state[:3] = (
                new_trajectory["finger_0"]["position"][-1].cpu().clone().numpy()
            )
            safe_state[2] = safe_trifinger_height
            safe_state[3:6] = (
                new_trajectory["finger_1"]["position"][-1].cpu().clone().numpy()
            )
            safe_state[5] = safe_trifinger_height
            trifinger_lcm.execute_trajectory(safe_state, no_data=True)

            # Add data to dataset
            add_trajectory = TensorDict({}, batch_size = new_trajectory.batch_size)
            add_trajectory["robot_state"] = extract_robot_trajectory(learned_system, new_trajectory, robot_model_name)
            for finger_name in new_trajectory.keys():
                try:
                    add_trajectory["contact_forces", finger_name] = new_trajectory[finger_name]["contact_force_W"]
                    add_trajectory["contact_normals", finger_name] = new_trajectory[finger_name]["contact_normal_W"]
                except (IndexError, KeyError): # e.g. object, time
                    continue
            add_trajectory["time"] = new_trajectory["time"]
            add_trajectory[object_model_name + "_groundtruth"] = new_trajectory[object_model_name]["position"]
            data_trajectories.add_trajectories(
                [add_trajectory.clone().detach()],
                torch.tensor([len(data_trajectories.trajectories)], dtype=torch.int),
            )

            # Simulate and Extend Learnable Trajectory
            # TODO: HACK Sim causes things to go flying, debug
            """
            print("Simulating init trajectory")
            with torch.no_grad():
                plant_states_dict, _, _ = learned_system.diff_simulate(
                    add_trajectory["robot_state"].unsqueeze(0), add_trajectory["time"],
                    steps_per_timestep=100
                )
            # TODO: HACK don't hardcode object model name
            learned_system.add_trajectories(
                traj_lens=[len(plant_states_dict.squeeze())],
                traj_data=[plant_states_dict.squeeze()["cube_state"]],
            )
            """
            learned_system.add_trajectories(
                traj_lens=[len(add_trajectory["time"])],
            )

            # Re-init visualizer
            gui_vis.update()

            # Re-init optimizer and data-loader
            batch_size = (
                len(data_trajectories.slices)
                if data_trajectories.slices.config.batch_size == -1
                else data_trajectories.slices.config.batch_size
            )
            traj_dataloader = DataLoader(
                data_trajectories.slices,
                batch_size=batch_size,
                shuffle=data_trajectories.slices.config.shuffle,
                generator=torch.Generator(device=torch.get_default_device()),
            )
            optimizer = optimizer_cls(learned_system.parameters())
            obs_info_inv = None

        elif command_char == "s":
            try:
                action = int(input("Which Action (<0 == random)? "))
            except ValueError:
                print("Cancelling...")
                continue
            if action < 0:
                temp = random.choice(action_library)
                print(f"Sampling random action: {temp}")
                selected_action = sample_action(library=temp)
            else:
                print(f"Selecting Action: {action_library[action % len(action_library)]}")
                selected_action = sample_action(library=action_library[action % len(action_library)])

        elif command_char == "a":
            print("Recording Inverse Observed Info")
            if obs_info_inv is None:
                obs_info = learned_system.observed_info(traj_dataloader, get_loss_args)
                obs_info_inv = torch.linalg.inv(obs_info + 1e-1 * torch.eye(obs_info.size()[-1]))

            print(f"Previously Observed Info: {torch.diagonal(obs_info)}")

            print(f"Sampling {len(action_library)} actions to optimize...")
            libaction = ActionLibrary.NONE
            action_samples = torch.stack([
                torch.vstack([torch.from_numpy(action).clone().to(torch.get_default_device()) for action in sample_action(library=libaction)])
                #for _ in range(n_actions_optimized)
                for libaction in action_library
            ])
            interpolated_actions, timestamps = interpolate_sampled_action(action_samples)
            robot_trajectories = extract_robot_trajectory(learned_system, interpolated_actions, robot_model_name)
            fishers = learned_system.expected_fisher_info(robot_trajectories, timestamps, robot_model_name)
            fishers_obs_weighted = torch.matmul(fishers + 0 * torch.eye(fishers.size()[-1]), obs_info_inv)

            fishers_singvals = torch.linalg.svdvals(fishers_obs_weighted).real
            # sum the smaller singular values
            fishers_traces = fishers_singvals[..., 1:].sum(dim=-1)

            best_action = action_samples[torch.argmax(fishers_traces)]
            print(f"Fisher Traces:")
            for action, trace in zip(action_library, fishers_traces):
                print(f"{action} : {trace}")
            print(f"Best Action: {action_library[torch.argmax(fishers_traces)]}")
            selected_action = (best_action[0, :].detach().cpu().numpy(), best_action[1, :].detach().cpu().numpy())

        elif command_char == "o":
            if traj_dataloader is None or len(traj_dataloader) == 0:
                print("Data required for observed info\n")
                continue

            obs_info = learned_system.observed_info(traj_dataloader, get_loss_args)
            obs_info_inv = torch.linalg.inv(obs_info + 1e0 * torch.eye(obs_info.size()[-1]))

        elif command_char == "v":
            print("Visualizing entire trajectory.")
            gui_vis.sweep()
            print("Done!")

        elif command_char == "t":
            if traj_dataloader is None or len(traj_dataloader) == 0:
                print("Cannot train without data.\n")
                continue

            try:
                epochs = int(input("How many epochs? "))
            except ValueError:
                print("Cancelling...")
                continue
            print("Training...")

            ### TODO: HACK don't repeat vis code
            # TODO: HACK don't hardcode object name
            object_name = "cube"
            true_pose = np.array([1., 0., 0., 0., 0., 0., 0.])
            if new_trajectory is not None and len(new_trajectory) >= 1:
                true_pose = new_trajectory[object_name]["position"][-1].detach().cpu().numpy()

            start_time = time.time()
            for idx in range(epochs):
                train_loss, loss_data = train_epoch(traj_dataloader, learned_system, optimizer)
                total_epochs += 1
                print(total_epochs, 
                    f"Loss (J): {train_loss:.3e};", 
                    f"Pred (Nm): {loss_data['mean_pred_Nm']:.3e};", 
                    f"Pred (<m=rad>/s): {loss_data['mean_q_pred_mps']:.3e};", 
                    f"Comp (Nm): {loss_data['mean_comp_Nm']:.3e};", 
                    f"Pen (m): {loss_data['mean_pen_m']:.3e};", 
                    f"Diss (J/s): {loss_data['mean_diss_Jps']:.3e};", 
                    f"Dev (N): {loss_data['mean_dev_N']:.3e};",
                    f"Norm (cosine): {loss_data['mean_norm_cosine']:.3e};",
                )
                gui_vis.update()
                train_losses.append(train_loss)
                train_loss_data.append(loss_data)
                learned_summaries.append(learned_system.summary({}))
                if signal_pressed:
                    signal_pressed = False
                    print("Training cancelled...")
                    epochs = idx + 1
                    break

            print(f"Finished training {epochs} epochs in {time.time()-start_time} seconds!")
            obs_info_inv = None

    # Quit


def main_fn():
    """Entry point"""
    config_file = DEFAULT_CONFIG
    if len(sys.argv) < 2:
        print(f"Warning: Using default config file ({DEFAULT_CONFIG})")
    else:
        config_file = sys.argv[1]

    # Parse config file and start
    gin.parse_config_file(os.path.join(REPO_DIR, "config", config_file))
    # Pylint doesn't know about gin
    # pylint: disable=no-value-for-parameter
    main()


if __name__ == "__main__":
    main_fn()
