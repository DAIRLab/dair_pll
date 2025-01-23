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

import os
import pdb
import sys
import time
from typing import cast, Any, Dict, List, Type, Tuple, Optional

import gin
#import gin.torch.external_configurables
import git
import lcm
import numpy as np
from scipy.spatial.transform import Rotation as R
from tensordict import TensorDictBase, TensorDict
import torch

# tensor_utils required for TensorDict's collate_fn
# pylint: disable-next=unused-import
from dair_pll import tensor_utils
from dair_pll.state_space import CenteredSampler
from dair_pll.lcmtypes.dairlib import lcmt_fingertips_position, lcmt_object_state, lcmt_densetact_measurement, lcmt_densetact_measurement_data, lcmt_fingertips_target_kinematics


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
    def __init__(self, lcm_channels: Dict[str, str], fingertip_body_names: List[str], traj_time_len=2.0):
        self._lcm_channels = lcm_channels
        self._traj_time_len = traj_time_len
        self._fingertip_body_names = fingertip_body_names

        self._force_raw_data = []
        self._fingertip_pose_raw_data = []
        self._object_raw_data = []

        # Init LCM Subscriptions
        self._lcm = lcm.LCM()
        self._lcm_subs = {}
        self._lcm_subs["fingertips_position"] = self._lcm.subscribe(lcm_channels["fingertips_position"], self.sub_handler)
        self._lcm_subs["densetact"] = self._lcm.subscribe(lcm_channels["densetact"], self.sub_handler)
        self._lcm_subs["object_state"] = self._lcm.subscribe(lcm_channels["object_state"], self.sub_handler)
        for sub in self._lcm_subs.values():
            sub.set_queue_capacity(1) # to discard everything outside of the handle window

    def sub_handler(self, channel: str, data: Any):
        if channel == self._lcm_channels["fingertips_position"]:
            self._fingertip_pose_raw_data.append(lcmt_fingertips_position.decode(data))
        if channel == self._lcm_channels["densetact"]:
            self._force_raw_data.append(lcmt_densetact_measurement_data.decode(data))
        if channel == self._lcm_channels["object_state"]:
            self._object_raw_data.append(lcmt_object_state.decode(data))

    def execute_trajectory(self, target_state: np.ndarray, pos_is_absolute: bool = True, no_data: bool = False) -> TensorDictBase:
        """
        Direct the robot to go to target_state.
        Record all incoming data over the next traj_time_len seconds.

        NOTE: assumes that target_state is in order (finger_0q, finger_120q, finger_240q, finger_0v, finger_120v, finger_240v)
        """
        command = lcmt_fingertips_target_kinematics()
        assert target_state.shape == (len(command.targetPos)+len(command.targetVel),)
        command.utime = int(time.time() * 1e6)
        command.isAbsoluteTargetPos = pos_is_absolute
        command.targetPos[:] = target_state[:len(command.targetPos)]
        command.targetVel[:] = target_state[len(command.targetPos):]

        print(f"Sending Command at: {time.time()}")
        self._lcm.publish(self._lcm_channels["fingertips_target"], command.encode())
        end_time = time.time() + self._traj_time_len
        while time.time() < end_time:
            self._lcm.handle_timeout(int((end_time-time.time()) * 1e3))
        print(f"Finished at: {time.time()}")
        print(f"Collected {len(self._fingertip_pose_raw_data)} / {len(self._force_raw_data)} / {len(self._object_raw_data)} samples.")

        # Return empty if not any force data
        ret = TensorDict({}, batch_size = len(self._force_raw_data))
        if no_data or len(self._force_raw_data) < 1:
            return ret

        assert self._force_raw_data[0].numSensors == len(self._fingertip_body_names)
        assert len(self._fingertip_pose_raw_data) >= len(self._force_raw_data)
        is_sorted = lambda a: np.all(a[:-1] <= a[1:])
        densetact_time_s = np.array([float(measurement.sensorData[0].timestamp)/1e6 for measurement in self._force_raw_data]).flatten()
        assert is_sorted(densetact_time_s)
        fingerpos_time_s = np.array([float(measurement.utime)/1e6 for measurement in self._fingertip_pose_raw_data]).flatten()
        assert is_sorted(fingerpos_time_s)
        object_time_s = np.array([float(measurement.utime)/1e6 for measurement in self._object_raw_data]).flatten()
        assert is_sorted(object_time_s)

        # Interp fingertip data
        fingertip_pos_W = {}
        fingertip_vel_W = {}
        fingertip_force_C = {}
        fingertip_normal_W = {}
        for body_idx, body_name in enumerate(self._fingertip_body_names):
            # Position Interpolation
            body_pos = np.array([measurement.curPos[3*body_idx:3*body_idx+3] for measurement in self._fingertip_pose_raw_data])
            assert body_pos.shape == (len(fingerpos_time_s), 3)
            body_pos_interp = np.vstack([np.interp(densetact_time_s, fingerpos_time_s, body_pos[:, idx]) for idx in range(3)]).T
            assert body_pos_interp.shape == (len(densetact_time_s), 3)
            fingertip_pos_W[body_name] = body_pos_interp

            # Velocity Interpolation
            body_vel = np.array([measurement.curVel[3*body_idx:3*body_idx+3] for measurement in self._fingertip_pose_raw_data])
            assert body_vel.shape == (len(fingerpos_time_s), 3)
            body_vel_interp = np.vstack([np.interp(densetact_time_s, fingerpos_time_s, body_vel[:, idx]) for idx in range(3)]).T
            assert body_vel_interp.shape == (len(densetact_time_s), 3)
            fingertip_vel_W[body_name] = body_vel_interp

            # Quat Interpolation
            body_quat = np.array([measurement.curQuat[3*body_idx:3*body_idx+4] for measurement in self._fingertip_pose_raw_data])
            assert body_quat.shape == (len(fingerpos_time_s), 4)
            body_quat_interp = np.vstack([np.interp(densetact_time_s, fingerpos_time_s, body_quat[:, idx]) for idx in range(4)]).T
            assert body_quat_interp.shape == (len(densetact_time_s), 4)
            body_R_BW = R.from_quat(body_quat_interp)
            
            # Record normal and force in world frame
            try:
                body_R_CB = R.from_matrix(np.stack([np.array(measurement.sensorData[body_idx].contactFrame)[:3,:3] for measurement in self._force_raw_data]))
                normal_C = np.broadcast_to(np.array([0., 0., 1.]), (len(densetact_time_s), 3))
                body_R_CW = body_R_BW.inv() * body_R_CB
                fingertip_normal_W[body_name] = body_R_CW.apply(normal_C)
                force_C = np.array([(list(measurement.sensorData[body_idx].scaledFriction) + [measurement.sensorData[body_idx].scaledNormal]) for measurement in self._force_raw_data])
                assert force_C.shape == (len(densetact_time_s), 3)
                fingertip_force_C[body_name] = force_C
            except ValueError:
                breakpoint()

        
        ret["time"] = torch.from_numpy(densetact_time_s)
        for body_name in self._fingertip_body_names:
            ret[body_name, "position"] = torch.from_numpy(fingertip_pos_W[body_name]).clone()
            ret[body_name, "velocity"] = torch.from_numpy(fingertip_vel_W[body_name]).clone()
            ret[body_name, "contact_force_C"] = torch.from_numpy(fingertip_force_C[body_name]).clone()
            ret[body_name, "contact_normal_W"] = torch.from_numpy(fingertip_normal_W[body_name]).clone()

        # Interp ground-truth object data
        if len(self._object_raw_data) > 0:
            # Position Interpolation
            num_positions = self._object_raw_data[0].num_positions
            object_pos = np.array([measurement.position[:] for measurement in self._object_raw_data])
            assert object_pos.shape == (len(object_time_s), num_positions)
            object_pos_interp = np.vstack([np.interp(densetact_time_s, object_time_s, object_pos[:, idx]) for idx in range(num_positions)]).T
            assert object_pos_interp.shape == (len(densetact_time_s), num_positions)
            ret[self._object_raw_data[0].object_name, "position"] = torch.from_numpy(object_pos_interp).clone()

            # Velocity Interpolation
            num_velocities = self._object_raw_data[0].num_velocities
            object_vel = np.array([measurement.velocity[:] for measurement in self._object_raw_data])
            assert object_vel.shape == (len(object_time_s), num_velocities)
            object_vel_interp = np.vstack([np.interp(densetact_time_s, object_time_s, object_vel[:, idx]) for idx in range(num_velocities)]).T
            assert object_vel_interp.shape == (len(densetact_time_s), num_velocities)
            ret[self._object_raw_data[0].object_name, "velocity"] = torch.from_numpy(object_vel_interp).clone()


        # Clear data and return
        self._force_raw_data.clear()
        self._fingertip_pose_raw_data.clear()
        self._object_raw_data.clear()
        return ret

@gin.configurable
def sample_action(workspace_xy_center: Tuple[float, float], workspace_z_rot: float, workspace_radius: float, sphere_radius: float, fixed_240_W: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a straight line action
    Params:
    workspace_xy_center: offset the workspace from world origin
    workspaxe_z_rot: world rotation so finger_0 is towards +X axis
    workspace_radius: start will be along edge of radius above ground
    sphere_radius: radius of robot fingertip
    fixed_240: 3d position 
    """

    assert workspace_radius > 0.
    assert sphere_radius > 0. and sphere_radius < workspace_radius
    assert len(fixed_240_W) == 3
    fixed_240_traj = np.array(fixed_240_W)
    rng = np.random.default_rng()

    # Start in workspace frame
    def sample_finger(flip_x: bool = False):
        flip_factor = -1.0 if flip_x else 1.0
        start_polar = rng.uniform(0., np.pi/2.0)
        start_azimuth = rng.uniform(-np.pi/2.0, np.pi/2.0)
        start_S = (workspace_radius-sphere_radius) * np.array([(np.sin(start_polar)*np.cos(start_azimuth)), np.sin(start_polar)*np.sin(start_azimuth), sphere_radius + np.cos(start_polar)])
        start_S[0] += sphere_radius
        start_S[0] *= flip_factor
        end_radius = rng.uniform(0., workspace_radius-sphere_radius)
        end_angle = rng.uniform(0., np.pi)
        end_S = np.array([flip_factor * sphere_radius, end_radius * np.cos(end_angle), sphere_radius + end_radius * np.sin(end_angle)])
        return (start_S, end_S)

    finger_0_traj = sample_finger(False)
    finger_120_traj = sample_finger(True)

    ret = (np.zeros(18), np.zeros(18))
    z_rot = np.array([[np.cos(workspace_z_rot), -np.sin(workspace_z_rot), 0.],
        [np.sin(workspace_z_rot), np.cos(workspace_z_rot), 0.],
        [0., 0., 1.]])
    xy_trans = np.array([workspace_xy_center[0], workspace_xy_center[1], 0.])
    for idx in [0, 1]:
        ret[idx][:3] = z_rot @ finger_0_traj[idx].T + xy_trans
        ret[idx][3:6] = z_rot @ finger_120_traj[idx].T + xy_trans
        ret[idx][6:9] = fixed_240_traj[:]
    return ret



## Main Function
@gin.configurable
def main(init_trifinger_state: List[float], safe_trifinger_height: float):
    """Main function for online learning loop"""
    trifinger_lcm = TrifingerLCMService()

    print("Move to initial trifinger state")
    trifinger_lcm.execute_trajectory(np.array(init_trifinger_state), no_data=True)

    print("Sample Initial Random Action...")
    selected_action = sample_action()

    # Start Input Loop
    def print_help():
        print(
            "\nUsage:\n"
            "e - Execute selected action + collect data\n"
            "s - Sample random action\n"
            "b - breakpoint()\n"
            "h - Print Help\n"
            "q - Quit\n"
        )

    print_help()
    command_char = " "
    while command_char != "q":
        command_char = input("Command $ ").split(" ")[0]

        if command_char == 'h':
            print_help()

        elif command_char == "b":
            # pylint: disable-next=forgotten-debug-statement
            pdb.Pdb(nosigint=True).set_trace()

        elif command_char == "e":
            # Move to start state
            trifinger_lcm.execute_trajectory(selected_action[0], no_data=True)

            # Execute and collect data
            new_data = trifinger_lcm.execute_trajectory(selected_action[1])

            # Move straight up
            safe_state = np.copy(selected_action[0])
            safe_state[:3] = new_data["finger_0"]["position"][-1].cpu().clone().numpy()
            safe_state[2] = safe_trifinger_height
            safe_state[3:6] = new_data["finger_1"]["position"][-1].cpu().clone().numpy()
            safe_state[5] = safe_trifinger_height
            trifinger_lcm.execute_trajectory(safe_state, no_data=True)

        elif command_char == "s":
            print("Sampling random action...")
            selected_action = sample_action()

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
    main()


if __name__ == "__main__":
    main_fn()
