"""Controller System Builders

Functions passed to `additional_system_builders` in order to create
control loops.
"""

from __future__ import annotations

import gin
import numpy as np

from pydrake.multibody.plant import MultibodyPlant  # type: ignore
from pydrake.systems.framework import DiagramBuilder  # type: ignore
from pydrake.systems.controllers import PidController
from pydrake.systems.framework import Context  # type: ignore
from pydrake.systems.primitives import (
    ConstantVectorSource,
    TrajectorySource,
    Multiplexer,
)
from pydrake.trajectories import Trajectory, PiecewisePolynomial


@gin.configurable(denylist=['system'])
def update_pid_reference(
    system: DrakeSystem,
    desired_state: np.ndarray,
    reference_name: str = "pid_reference"
):
    # Get underlying drake system and context
    pid_system = system.plant_diagram.diagram.GetSubsystemByName(reference_name)
    sim_context = system.plant_diagram.sim.get_mutable_context()
    pid_context = pid_system.GetMyContextFromRoot(sim_context)

    # Update mutable vector source
    vector_source = pid_system.get_mutable_source_value(pid_context)
    assert desired_state.size == pid_system.get_output_port().size()

    vector_source.set_value(desired_state)


@gin.configurable(denylist=['builder', 'plant'])
def pid_controller_builder(
    builder: DiagramBuilder,
    plant: MultibodyPlant,
    reference_name: str = "pid_reference",
    desired_state: Optional[List[float]] = None,
    model_name: str = "robot",
    kp: float = 1.0,
    kd: float = 10.0,
):
    model = plant.GetModelInstanceByName(model_name)
    control_size = plant.get_actuation_input_port(model).size() 

    if desired_state is None:
        desired_state = np.zeros(2*control_size)
    else:
        desired_state = np.array(desired_state)

    assert desired_state.size == 2*control_size # Note assumes n_q == n_v

    controller = PidController(
        kp * np.ones(control_size), np.zeros(control_size), kd * np.ones(control_size)
    )
    

    controller = builder.AddSystem(controller)
    builder.Connect(
        plant.get_state_output_port(model), controller.get_input_port_estimated_state()
    )
    builder.Connect(
        controller.get_output_port_control(), plant.get_actuation_input_port(model)
    )

    # Desired State
    constant = ConstantVectorSource(desired_state)
    constant.set_name(reference_name)
    constant = builder.AddSystem(constant)
    builder.Connect(
        constant.get_output_port(), controller.get_input_port_desired_state()
    )


def traj_follower_builder(
    builder: DiagramBuilder,
    plant: MultibodyPlant,
    trajectory: Trajectory,
    model_name: str = "robot",
    kp: float = 1.0,
    kd: float = 10.0,
):
    traj_source = TrajectorySource(trajectory)
    derivative = trajectory.derivative()
    deriv_source = TrajectorySource(derivative)
    q_size = trajectory.value(0.0).size
    v_size = derivative.value(0.0).size
    assert q_size == v_size
    controller = PidController(
        kp * np.ones(q_size), np.zeros(q_size), kd * np.zeros(v_size)
    )
    model = plant.GetModelInstanceByName(model_name)

    # Add Conroller to graph
    controller = builder.AddSystem(controller)
    builder.Connect(
        plant.get_state_output_port(model), controller.get_input_port_estimated_state()
    )
    builder.Connect(
        controller.get_output_port_control(), plant.get_actuation_input_port(model)
    )

    # Desired state from trajectory
    traj_source = builder.AddSystem(traj_source)
    deriv_source = builder.AddSystem(deriv_source)
    traj_multiplex = builder.AddSystem(Multiplexer([q_size, v_size]))
    builder.Connect(traj_source.get_output_port(), traj_multiplex.get_input_port(0))
    builder.Connect(deriv_source.get_output_port(), traj_multiplex.get_input_port(1))
    builder.Connect(
        traj_multiplex.get_output_port(), controller.get_input_port_desired_state()
    )


def traj_with_knots_follower_builder(
    builder: DiagramBuilder,
    plant: MultibodyPlant,
    traj_breaks: np.ndarray,
    traj_q: np.ndarray,
    traj_v: Optional[np.ndarray] = None,
    model_name: str = "robot",
    kp: float = 1.0,
    kd: float = 10.0,
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
