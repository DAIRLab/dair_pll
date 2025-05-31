#!/usr/bin/env python3

"""Utility functions for GUI visualizations (including Meshcat)

The main contents of this file are as follows:

    * A class to initatite a comparison visualization in Meshcat between the true and learned trajectory + geometry
"""

import numpy as np
from pydrake.geometry import StartMeshcat, Meshcat, Shape, Rgba
from scipy.spatial.transform import Rotation as R
from tkinter import Tk, Scale, DoubleVar
import torch
import time

from dair_pll.multibody_learnable_system import MultibodyLearnableSystemWithTrajectory
from dair_pll.dataset_management import TrajectorySet


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
        system: MultibodyLearnableSystemWithTrajectory | MultibodyLearnableTactileSystem,
        data: TrajectorySet,
        true_geom: Shape,
    ) -> None:
        self._meshcat = StartMeshcat()
        self._data = data
        self._system = system
        self._meshcat.SetObject("/true", true_geom, Rgba(0.8, 0.0, 0.0, 0.3))
        self._meshcat.SetTransform("/true", transform_from_state_q(np.array([1., 0., 0., 0., 0., 0., 0.])))
        self._meshcat.SetObject("/learned", self._system.get_learned_geometry(), Rgba(0.0, 0.0, 0.8, 1.0))
        self._meshcat.SetTransform("/learned", transform_from_state_q(self._system.get_learned_pose().cpu().numpy()))

        self._learned_plant_traj = None

        self.reinit_tk()

  @property
  def learned_plant_traj(self):
    return self._learned_plant_traj

  @learned_plant_traj.setter
  def learned_plant_traj(self, value):
    self._learned_plant_traj = value.detach()
  

  def reinit_tk(self, new_val=0.) -> None:
    self._root = Tk()
    self._timestep = DoubleVar(value=new_val)
    self._scale = Scale(master=self._root, 
      variable=self._timestep, 
      digits=0, 
      from_=0., 
      to=new_val,
      length=900,
      orient="horizontal",
      resolution=1.,
      command=self.update)
    self._scale.pack()

  def update(self, event=None) -> None:
    """
    Update visualization
    """
    # Update learned geometry
    self._meshcat.SetObject("/learned", self._system.get_learned_geometry(), Rgba(0.0, 0.0, 0.8, 1.0))

    # Update learned trajectory
    timestep = int(self._timestep.get())
    learned_traj = self._system.get_learned_trajectory(self._learned_plant_traj).cpu().numpy() # (traj_len, 7)
    self._meshcat.SetTransform("/learned", transform_from_state_q(learned_traj[timestep, :]))

    # Update true trajectory
    # TODO: HACK remove hardcoding
    if len(self._data.trajectories) > 0:
      true_traj = torch.cat([traj["cube_groundtruth"] for traj in self._data.trajectories], dim=-2).cpu().numpy()
      assert true_traj.shape == learned_traj.shape
      self._meshcat.SetTransform("/true", transform_from_state_q(true_traj[timestep, :]))
      self._scale.configure(to=true_traj.shape[0]-1)

      ## Robot
      self._meshcat.SetObject("/robot/0", self._system.get_body_geometry("finger_0"), Rgba(0.8, 0.0, 0.0, 0.8))
      self._meshcat.SetObject("/robot/1", self._system.get_body_geometry("finger_1"), Rgba(0.8, 0.0, 0.0, 0.8))
      robot_traj = torch.cat([traj["robot_state"] for traj in self._data.trajectories], dim=-2).cpu().numpy()
      zero_rot = np.array([1., 0., 0., 0.])
      robot_0_traj = np.hstack([np.broadcast_to(zero_rot, (robot_traj.shape[0], 4)), robot_traj[:, :3]])
      robot_1_traj = np.hstack([np.broadcast_to(zero_rot, (robot_traj.shape[0], 4)), robot_traj[:, 3:6]])
      self._meshcat.SetTransform("/robot/0", transform_from_state_q(robot_0_traj[timestep, :]))
      self._meshcat.SetTransform("/robot/1", transform_from_state_q(robot_1_traj[timestep, :]))
      
  def sweep(self, dt=0.033) -> None:
    """
    Sweep through the entire trajectory
    """
    end = int(self._scale.config()['to'][-1])

    for timestep in range(end):
      self._timestep.set(timestep)
      self.update()
      time.sleep(dt)

