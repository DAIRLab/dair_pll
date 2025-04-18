import os
import git 

import numpy as np
import gin
from typing import Dict, List, Any
from tensordict import TensorDictBase, TensorDict
import lcm
import time
import torch

from dair_pll.lcmtypes.dairlib import (
    lcmt_fingertips_position,
    lcmt_object_state,
    lcmt_densetact_measurement_data,
    lcmt_fingertips_target_kinematics,
)

from scipy.spatial.transform import Rotation as R

## Execute Robot Trajectory
@gin.configurable
class TrifingerLCMService:
    """
    Command robot and collect data over LCM
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self,
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

    def __exit__(self):
        del self._lcm

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