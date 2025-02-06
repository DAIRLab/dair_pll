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

import os
import pdb
import sys
import time
from typing import Any, Dict, List, Tuple

import gin

# import gin.torch.external_configurables
import git
import lcm
import numpy as np
from scipy.spatial.transform import Rotation as R
from tensordict import TensorDictBase, TensorDict
import torch
from torch import Tensor

from scipy.spatial import geometric_slerp

# tensor_utils required for TensorDict's collate_fn
# pylint: disable-next=unused-import
from dair_pll import tensor_utils
from dair_pll import file_utils
from dair_pll.multibody_learnable_system import MultibodyLearnableSystemWithTrajectory
from dair_pll.lcmtypes.dairlib import (
    lcmt_fingertips_position,
    lcmt_object_state,
    lcmt_densetact_measurement_data,
    lcmt_fingertips_target_kinematics,
)
from dair_pll.hack_utils import finger_idx_from_body_name

# Repository directory (default for file operations)
REPO_DIR = os.path.normpath(
    git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")
)
DEFAULT_CONFIG = "rss_experiment.gin"


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
        traj_time_len = 2.0,
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

    def traj_interpolation(self,
        init_state: np.ndarray, 
        target_state: np.ndarray,
        steps: int = 100):

        dt = self._traj_time_len/steps
        command = lcmt_fingertips_target_kinematics()

        # num x 18 array
        init_state = init_state[:9]
        target_state = target_state[:9]

        traj_interp_lin = np.linspace(init_state, target_state, num = steps)

        command.utime = int(time.time() * 1e6)
        command.isAbsoluteTargetPos = True
        command.targetPos[:] = target_state
        command.targetVel[:] = np.zeros_like(target_state)

        self._lcm.publish(self._lcm_channels["fingertips_target"], command.encode())

        start_time = time.time()
        while time.time() < start_time + self._traj_time_len:
            self._lcm.handle_timeout(int((start_time + self._traj_time_len - time.time()) * 1e3))



        # traj_interp_circ = geometric_slerp(init_state/np.linalg.norm(init_state), target_state/np.linalg.norm(target_state), t = np.linspace(0,1, steps))
        # mag = np.linspace(np.linalg.norm(init_state), np.linalg.norm(target_state), num=steps)
        # mag *= np.ones_like(mag) + 1.25*np.sin(np.linspace(0, np.pi, steps))**2
        # traj_interp = np.multiply(traj_interp_circ, mag[:, np.newaxis])


        #print(np.round(traj_interp_circ,2), np.round(traj_interp,2))

        # violation = np.any(traj_interp_lin < 0.025, axis=0) & np.any(traj_interp_lin > -0.025, axis=0)
        # print(violation)

        # for waypoint_p in np.split(traj_interp_lin, steps, axis = 0):
        #     #print(waypoint_p)
        #     start_time = time.time()

        #     command.utime = int(time.time() * 1e6)
        #     command.isAbsoluteTargetPos = True
        #     command.targetPos[:] = waypoint_p[0]
        #     command.targetVel[:] = np.zeros_like(waypoint_p)[0]

        #     self._lcm.publish(self._lcm_channels["fingertips_target"], command.encode())

        #     while time.time() < start_time + dt:
        #         self._lcm.handle_timeout(int(dt * 1e3))



    def execute_trajectory(
        self,
        target_state: np.ndarray,
        pos_is_absolute: bool = True,
        no_data: bool = False,
        no_collision: bool = False
    ) -> TensorDictBase:
        """
        Direct the robot to go to target_state.
        Record all incoming data over the next traj_time_len seconds.

        NOTE: assumes that target_state is in order 
            (finger_0q, finger_120q, finger_240q, finger_0v, finger_120v, finger_240v)
        """
        # pylint: disable=too-many-locals



        print(f"Sending Command at: {time.time()}")
        # if no_collision:
        #     command.targetPos[:]
        #     self._lcm.publish(self._lcm_channels["fingertips_target"], command.encode())
        # else:
        #     self._lcm.publish(self._lcm_channels["fingertips_target"], command.encode())
        self._lcm.handle()

        trifinger_state = self._fingertip_pose_raw_data
        init_state = np.concatenate([np.array(trifinger_state[-1].curPos), np.array(trifinger_state[-1].curVel)])

        self.traj_interpolation(init_state, target_state, 50)



        # end_time = time.time() + self._traj_time_len

        # while time.time() < end_time:
        #     self._lcm.handle_timeout(int((end_time - time.time()) * 1e3))
        print(f"Finished at: {time.time()}")
        print(
            f"Collected {len(self._fingertip_pose_raw_data)}" +
            f"/ {len(self._force_raw_data)} / {len(self._object_raw_data)} samples."
        )

        # Return empty if not any force data
        ret = TensorDict({}, batch_size=len(self._force_raw_data))
        if no_data or len(self._force_raw_data) < 1:
            self._force_raw_data.clear()
            self._fingertip_pose_raw_data.clear()
            self._object_raw_data.clear()
            return ret

        assert self._force_raw_data[0].numSensors == len(self._fingertip_body_names)

        assert len(self._fingertip_pose_raw_data) >= len(self._force_raw_data)

        def is_sorted(a: np.ndarray) -> bool:
            return np.all(a[:-1] <= a[1:])
        densetact_time_s = np.array(
            [
                float(measurement.sensorData[0].utime) / 1e6
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
        fingertip_normal_W = {}

        # enumerate each fingertip ie 1,2,3 with its name
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
                    measurement.curQuat[3 * body_idx : 3 * body_idx + 4]
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
            body_R_BW = R.from_quat(body_quat_interp)

            # Record normal and force in world frame
            body_R_CB = R.from_matrix(
                np.stack(
                    [
                        np.array(measurement.sensorData[body_idx].contactPose)[:3, :3]
                        for measurement in self._force_raw_data
                    ]
                )
            )
            normal_C = np.broadcast_to(
                np.array([0.0, 0.0, 1.0]), (len(densetact_time_s), 3)
            )
            body_R_CW = body_R_BW.inv() * body_R_CB
            fingertip_normal_W[body_name] = body_R_CW.apply(normal_C)
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

        # Clear data and return
        self._force_raw_data.clear()
        self._fingertip_pose_raw_data.clear()
        self._object_raw_data.clear()
        return ret


@gin.configurable
def sample_action(
    workspace_xy_center: Tuple[float, float],
    workspace_z_rot: float,
    workspace_radius: float,
    sphere_radius: float,
    fixed_240_W: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a straight line action
    Params:
    workspace_xy_center: offset the workspace from world origin
    workspaxe_z_rot: world rotation so finger_0 is towards +X axis
    workspace_radius: start will be along edge of radius above ground
    sphere_radius: radius of robot fingertip
    fixed_240: 3d position
    """

    assert workspace_radius > 0.0
    assert 0.0 < sphere_radius < workspace_radius
    assert len(fixed_240_W) == 3
    fixed_240_traj = np.array(fixed_240_W)
    rng = np.random.default_rng()

    # Start in workspace frame
    def sample_finger(flip_x: bool = False):
        flip_factor = -1.0 if flip_x else 1.0
        start_polar = rng.uniform(0.0, np.pi / 2.0)
        start_azimuth = rng.uniform(-np.pi / 2.0, np.pi / 2.0)
        start_S = (workspace_radius - sphere_radius) * np.array(
            [
                (np.sin(start_polar) * np.cos(start_azimuth)),
                np.sin(start_polar) * np.sin(start_azimuth),
                sphere_radius + np.cos(start_polar),
            ]
        )
        start_S[0] += sphere_radius
        start_S[0] *= flip_factor
        end_radius = rng.uniform(0.0, workspace_radius - sphere_radius)
        end_angle = rng.uniform(0.0, np.pi)
        end_S = np.array(
            [
                flip_factor * sphere_radius,
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
    Params:
        data: outputs of sample_action size (batch, 2, 18)

    Returns:
        TensorDict input of extract_robot_trajectory, batch_size=(batch, traj_n_steps), keys = fingertip_body_names
        Timestamps = Tensor size (batch, traj_n_steps)
    """
    assert len(data.size()) >= 2
    batch_dims = data.size()[:-2]
    assert data.size() == batch_dims + (2, 18)
    ret = TensorDict({}, batch_size=batch_dims + (traj_n_steps,))
    ret_timestamps = torch.linspace(0.0, traj_len_s, traj_n_steps)

    for fingertip in fingertip_body_names:
        ret[fingertip, "position"] = torch.zeros(batch_dims + (traj_n_steps, 3))
        ret[fingertip, "velocity"] = torch.zeros(batch_dims + (traj_n_steps, 3))

    for idx in range(traj_n_steps):
        data_lerp = torch.lerp(
            data[..., 0, :], data[..., 1, :], float(idx) / float(traj_n_steps)
        )
        for finger_idx, fingertip in enumerate(fingertip_body_names):
            pos_idx = 3 * finger_idx
            vel_idx = 9 + pos_idx
            ret[fingertip, "position"][..., idx, :] = data_lerp[..., pos_idx : pos_idx + 3]
            ret[fingertip, "velocity"][..., idx, :] = data_lerp[..., vel_idx : vel_idx + 3]

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


## Main Function
@gin.configurable
def main(
    init_trifinger_state: List[float],
    safe_trifinger_height: float,
    robot_model_name: str,
    storage_folder_name: str = "storage_rss",
    run_name: str = "default_run",
):
    """Main function for online learning loop"""
    #torch.autograd.set_detect_anomaly(True)
    #torch.set_default_device("cuda")

    # Create run directory
    print("Active Tactile Exploration")
    storage_name = os.path.join(REPO_DIR, "results", storage_folder_name)
    print(f"Storing data and results at {file_utils.run_dir(storage_name, run_name)}")

    # Initialize LCM
    # Pylint doesn't know about gin
    # pylint: disable=no-value-for-parameter
    trifinger_lcm = TrifingerLCMService()
    print("Move to initial trifinger state")
    trifinger_lcm.execute_trajectory(np.array(init_trifinger_state), no_data=True, no_collision = True)
    print("Sample Initial Random Action...")
    selected_action = sample_action(workspace_z_rot = np.pi/4)
    new_trajectory = None

    # Create learnable system
    print("Loading Learned System...")
    learned_system = MultibodyLearnableSystemWithTrajectory(
        output_urdfs_dir=file_utils.get_learned_urdf_dir(storage_name, run_name)
    )

    # Start Input Loop
    def print_help():
        print(
            "\nUsage:\n"
            "e - Execute selected action + collect data\n"
            "s - Sample random action\n"
            "t - Testing (reserved)\n"
            "b - breakpoint()\n"
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

            ## TODO: Add data to trajectory set
            

        elif command_char == "s":
            print("Sampling random action...")
            selected_action = sample_action()

        elif command_char == "t":
            print("Calculating Fisher Trace")
            torch_action = torch.vstack([torch.from_numpy(action).clone() for action in selected_action])
            interpolated_action, timestamps = interpolate_sampled_action(torch_action)
            robot_trajectory = extract_robot_trajectory(learned_system, interpolated_action, robot_model_name)

            trace = learned_system.trace_fisher_info(robot_trajectory.unsqueeze(0), timestamps, robot_model_name)

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
