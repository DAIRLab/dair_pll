"""Controller System Builders

Functions passed to `additional_system_builders` in order to create
control loops.
"""
from __future__ import annotations

import numpy as np

from pydrake.multibody.plant import MultibodyPlant  # type: ignore
from pydrake.systems.framework import DiagramBuilder # type: ignore
from pydrake.systems.controllers import PidController
from pydrake.systems.primitives import ConstantVectorSource, TrajectorySource, Multiplexer
from pydrake.trajectories import Trajectory, PiecewisePolynomial

def pid_controller_builder(builder: DiagramBuilder, 
    plant: MultibodyPlant, 
    desired_state: np.ndarray = np.zeros(2), 
    model_name: str = "robot", 
    kp: float = 1.,
    kd: float = 10.
):
    control_size = int(desired_state.size/2)
    controller = PidController(kp * np.ones(control_size), np.zeros(control_size), kd * np.zeros(control_size))
    model = plant.GetModelInstanceByName(model_name)

    controller = builder.AddSystem(controller)
    builder.Connect(
        plant.get_state_output_port(model),
        controller.get_input_port_estimated_state()
    )
    builder.Connect(
        controller.get_output_port_control(),
        plant.get_actuation_input_port(model)
    )

    # Desired State
    constant = ConstantVectorSource(desired_state)
    constant = builder.AddSystem(constant)
    builder.Connect(
        constant.get_output_port(),
        controller.get_input_port_desired_state()
    )

def traj_follower_builder(builder: DiagramBuilder, 
    plant: MultibodyPlant, 
    trajectory: Trajectory, 
    model_name: str = "robot",
    kp: float = 1.,
    kd: float = 10.
):
    traj_source = TrajectorySource(trajectory)
    derivative = trajectory.derivative()
    deriv_source = TrajectorySource(derivative)
    q_size = trajectory.value(0.).size
    v_size = derivative.value(0.).size
    assert q_size == v_size
    controller = PidController(kp * np.ones(q_size), np.zeros(q_size), kd * np.zeros(v_size))
    model = plant.GetModelInstanceByName(model_name)

    # Add Conroller to graph
    controller = builder.AddSystem(controller)
    builder.Connect(
        plant.get_state_output_port(model),
        controller.get_input_port_estimated_state()
    )
    builder.Connect(
        controller.get_output_port_control(),
        plant.get_actuation_input_port(model)
    )

    # Desired state from trajectory
    traj_source = builder.AddSystem(traj_source)
    deriv_source = builder.AddSystem(deriv_source)
    traj_multiplex = builder.AddSystem(Multiplexer([q_size, v_size]))
    builder.Connect(
        traj_source.get_output_port(),
        traj_multiplex.get_input_port(0)
    )
    builder.Connect(
        deriv_source.get_output_port(),
        traj_multiplex.get_input_port(1)
    )
    builder.Connect(
        traj_multiplex.get_output_port(),
        controller.get_input_port_desired_state()
    )

def traj_with_knots_follower_builder(builder: DiagramBuilder, 
    plant: MultibodyPlant, 
    traj_breaks: np.ndarray,
    traj_q: np.ndarray,
    traj_v: Optional[np.ndarray] = None,
    model_name: str = "robot",
    kp: float = 1.,
    kd: float = 10.
):
    # Input Sanitization
    assert len(traj_breaks) == len(traj_q)
    assert len(traj_q.shape) == 2
    model = plant.GetModelInstanceByName(model_name)
    num_q = len(plant.GetPositionNames(model))
    assert num_q == traj_q.shape[1]
    trajectory = PiecewisePolynomial.FirstOrderHold(traj_breaks, traj_q.T)

    if traj_v is not None:
        assert len(traj_breaks) == len(traj_v)
        assert len(traj_v.shape) == 2
        # State v size must match state q size
        num_v = len(plant.GetVelocityNames(model))
        assert num_v == traj_v.shape[1]

        trajectory = PiecewisePolynomial.CubicHermite(traj_breaks, traj_q.T, traj_v.T)

    return traj_follower_builder(builder, plant, trajectory, model_name, kp, kd)
