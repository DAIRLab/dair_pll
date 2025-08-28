#!/usr/bin/env python3

"""Utility functions for GUI visualizations (including Meshcat)

The main contents of this file are as follows:

        * A class to initatite a comparison visualization in Meshcat between the true and learned trajectory + geometry
"""

import time
from tkinter import Tk, Scale, DoubleVar
from typing import Optional

import numpy as np
from pydrake.geometry import StartMeshcat, Meshcat, Shape, Rgba
from scipy.spatial.transform import Rotation as R

from dair_pll.multibody_learnable_system import MultibodyLearnableSystemWithTrajectory
from dair_pll.multibody_tactile_learnable_system import MultibodyLearnableTactileSystem
from dair_pll.dataset_management import TrajectorySet
from dair_pll.hack_utils import finger_idx_from_body_name


def transform_from_state_q(state_q: np.ndarray):
    """
    Return a 4x4 transformation matrix from a len 7 state
    """
    assert state_q.shape == (7,)
    ret = np.eye(4)
    ret[:3, :3] = R.from_quat(state_q[:4], scalar_first=True).as_matrix()
    ret[:3, 3] = state_q[4:]
    return ret


class PLLMeshcatVisualizer:
    """
    Meshcat Comparison Visualization
    """

    _meshcat: Meshcat
    _system: MultibodyLearnableSystemWithTrajectory
    _data: TrajectorySet

    # Visualization
    _root: Tk
    _scale: Scale
    _timestep: DoubleVar

    def __init__(
        self,
        system: (
            MultibodyLearnableSystemWithTrajectory | MultibodyLearnableTactileSystem
        ),
        data: TrajectorySet,
        true_geom: Shape,
        true_pose: Optional[np.ndarray] = None,
    ) -> None:
        self._meshcat = StartMeshcat()
        self._data = data
        self._system = system
        self._meshcat.SetObject("/true", true_geom, Rgba(0.8, 0.0, 0.0, 0.3))

        true_transform = (
            true_pose
            if true_pose is not None
            else np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        )
        self._meshcat.SetTransform(
            "/true",
            transform_from_state_q(true_transform),
        )
        self._meshcat.SetObject(
            "/learned", self._system.get_learned_geometry(), Rgba(0.0, 0.0, 0.8, 1.0)
        )
        self._meshcat.SetTransform(
            "/learned",
            transform_from_state_q(self._system.get_learned_pose().cpu().numpy()),
        )

        self._learned_plant_traj = None

        self.reinit_tk()

    @property
    def learned_plant_traj(self):
        """Get optional learned full-plant trajectory"""
        return self._learned_plant_traj

    @learned_plant_traj.setter
    def learned_plant_traj(self, value):
        """Set optional learned full-plant trajectory"""
        self._learned_plant_traj = value.detach()

    def draw_action_samples(self, action_knots: np.ndarray):
        """Draw lines representing actions"""

        self.clear_action_samples()

        n_knots = action_knots.shape[0]
        assert action_knots.shape == (n_knots, 2, 18)

        # TODO: make gin-config param
        fingertip_body_names = ["finger_0", "finger_1"]

        for knot_idx in range(n_knots):
            for finger_idx, finger_name in enumerate(fingertip_body_names):
                str_key = f"/actions/{knot_idx}/{finger_name}"
                start_loc = action_knots[
                    knot_idx, 0, (finger_idx * 3) : ((finger_idx + 1) * 3)
                ]
                end_loc = action_knots[
                    knot_idx, 1, (finger_idx * 3) : ((finger_idx + 1) * 3)
                ]
                vertices = np.stack([start_loc, end_loc], axis=1)
                assert vertices.shape == (3, 2), str(vertices.shape)
                self._meshcat.SetLine(
                    path=str_key,
                    vertices=vertices,
                    line_width=4.0,
                    rgba=Rgba(r=0.1, g=0.9, b=0.9, a=1.0),
                )

    def clear_action_samples(self):
        self._meshcat.Delete("/actions")

    def reinit_tk(self, new_val=0.0) -> None:
        """Reset scale range to new value"""
        self._root = Tk()
        self._timestep = DoubleVar(value=new_val)
        self._scale = Scale(
            master=self._root,
            variable=self._timestep,
            digits=0,
            from_=0.0,
            to=new_val,
            length=900,
            orient="horizontal",
            resolution=1.0,
            command=self.update,
        )
        self._scale.pack()

    def update(self, _event=None) -> None:
        """
        Update visualization
        """
        # Update learned geometry
        self._meshcat.SetObject(
            "/learned", self._system.get_learned_geometry(), Rgba(0.0, 0.0, 0.8, 1.0)
        )

        # Update learned trajectory
        timestep = int(self._timestep.get())
        learned_traj = (
            self._system.get_learned_trajectory(self._learned_plant_traj).cpu().numpy()
        )  # (traj_len, 7)
        self._meshcat.SetTransform(
            "/learned", transform_from_state_q(learned_traj[timestep, :])
        )

        # Update true trajectory
        if len(self._data.trajectories) > 0 or self._learned_plant_traj is not None:
            if len(self._data.trajectories) > 0:
                true_traj = (
                    self._data.get_full_trajectory(
                        key=self._system.learned_model_names[0] + "_groundtruth"
                    )
                    .cpu()
                    .numpy()
                )
            else:
                true_traj = learned_traj
            assert true_traj.shape == learned_traj.shape
            self._meshcat.SetTransform(
                "/true", transform_from_state_q(true_traj[timestep, :])
            )
            self._scale.configure(to=true_traj.shape[0] - 1)

            ## Robot
            """
            self._meshcat.SetObject(
                "/robot/finger_0",
                self._system.get_body_geometry("finger_0"),
                Rgba(0.0, 0.8, 0.0, 0.8),
            )
            self._meshcat.SetObject(
                "/robot/finger_1",
                self._system.get_body_geometry("finger_1"),
                Rgba(0.0, 0.8, 0.0, 0.8),
            )
            robot_traj = (
                self._data.get_full_trajectory(
                    key=self._system.controlled_model_names[0] + "_state"
                )
                .cpu()
                .numpy()
            )
            """
            # TODO: make gin-config param
            fingertip_body_names = ["finger_0", "finger_1"]
            robot_traj = (
                self._system.get_controlled_trajectory(self._learned_plant_traj)
                .cpu()
                .numpy()
            )
            plant = self._system.plant
            robot_model_name = self._system.controlled_model_names[0]
            for finger_name, state_idx in zip(
                fingertip_body_names,
                finger_idx_from_body_name(
                    plant,
                    plant.GetModelInstanceByName(robot_model_name),
                    fingertip_body_names,
                ),
            ):
                if state_idx < 0:
                    continue
                pos_idx = 3 * state_idx
                body_pos_traj = robot_traj[..., pos_idx : pos_idx + 3]
                self._meshcat.SetObject(
                    f"/robot/{finger_name}",
                    self._system.get_body_geometry(finger_name),
                    Rgba(0.0, 0.8, 0.0, 0.8),
                )

                zero_rot = np.array([1.0, 0.0, 0.0, 0.0])
                body_traj = np.hstack(
                    [np.broadcast_to(zero_rot, (robot_traj.shape[0], 4)), body_pos_traj]
                )

                self._meshcat.SetTransform(
                    f"/robot/{finger_name}",
                    transform_from_state_q(body_traj[timestep, :]),
                )

            # Draw Contact Normals
            for body_name, normals in self._data.get_full_trajectory(
                key="contact_normals"
            ).items():
                str_key = f"/robot/{body_name}/normal"
                start_loc = np.zeros(3)
                # Negative normal to go into object
                end_loc = start_loc - 0.02 * normals.detach().cpu().numpy()[timestep]
                # print(f"Normal: {normals.detach().cpu().numpy()[timestep]}")
                vertices = np.stack([start_loc, end_loc], axis=1)
                assert vertices.shape == (3, 2), str(vertices.shape)
                # print(f"Drawing Normal: {normals.detach().cpu().numpy()[timestep]}")
                self._meshcat.SetLine(
                    path=str_key,
                    vertices=vertices,
                    line_width=4.0,
                    rgba=Rgba(r=0.9, g=0.1, b=0.9, a=1.0),
                )

    def sweep(self, dt=0.033) -> None:
        """
        Sweep through the entire trajectory
        """
        end = int(self._scale.config()["to"][-1])

        for timestep in range(end):
            self._timestep.set(timestep)
            self.update()
            time.sleep(dt)
