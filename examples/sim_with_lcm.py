#!/usr/bin/env python3
"""
Run a simulated with LCM interface to mimic hardware
"""
import os
from typing import cast, Any, Dict, List, Type, Optional
import sys

import gin
import git
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from pydrake.common.value import Value
from pydrake.lcm import DrakeLcm
from pydrake.math import RigidTransform
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.systems.controllers import PidController
from pydrake.systems.drawing import plot_system_graphviz
from pydrake.systems.framework import BasicVector, DiagramBuilder, LeafSystem, EventStatus
from pydrake.systems.lcm import LcmInterfaceSystem, LcmPublisherSystem, LcmSubscriberSystem, PySerializer
from pydrake.trajectories import PiecewisePolynomial

from dair_pll.drake_utils import MultibodyPlantDiagram, ContactForceAveragerLeafSystem, get_bodies_in_model_instance
from dair_pll.lcmtypes.dairlib import lcmt_fingertips_position, lcmt_object_state, lcmt_densetact_measurement, lcmt_densetact_measurement_data, lcmt_fingertips_target_kinematics
from dair_pll import file_utils
from dair_pll.hack_utils import finger_idx_from_body_name

# Repository directory (default for file operations)
REPO_DIR = os.path.normpath(
    git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")
)
DEFAULT_CONFIG = "sim_with_lcm.gin"

## Diagram Builder
@gin.configurable(denylist=['plant', 'robot_id'])
class SetpointFromTargetSystem(LeafSystem):
  """Create a Drake ``LeafSystem`` which converts an incoming target
  into a CubicSpline trajectory. Then outputs a setpoint given sim time."""

  def __init__(self, plant: MultibodyPlant, robot_id: ModelInstanceIndex, fingertip_body_names: List[str], traj_time_len: float = 1.0):
    super().__init__()

    self._body_names = fingertip_body_names
    self._plant = plant
    self._robot_id = robot_id
    self._traj_time_len = traj_time_len

    # Input for LCM Target Message
    self._target_kinematics_input_port = self.DeclareAbstractInputPort(
        "lcmt_fingertips_target_kinematics", Value(lcmt_fingertips_target_kinematics())
    )

    # Input for Robot State
    self._robot_state_input_port = self.DeclareVectorInputPort(
      "robot_state", BasicVector(self._plant.num_multibody_states(self._robot_id))
    )

    # Output setpoint state for controller
    self.DeclareVectorOutputPort("robot_setpoint",
         self._plant.num_multibody_states(robot_id),
         self.calc_setpoint)

    # Abstract state to hold CubicSpline trajectory
    self._trajectory_index = self.DeclareAbstractState(
            Value(PiecewisePolynomial())
        )

    # Update event to poll LCM subscriber
    self.DeclarePerStepUnrestrictedUpdateEvent(self.poll_lcm_target)

    self._prev_target_timestamp_idx = self.DeclareDiscreteState(np.ones(1) * -1)

  def calc_setpoint(self, context, robot_setpoint):
    self.ValidateContext(context)
    time = context.get_time()
    traj = context.get_abstract_state(self._trajectory_index).get_value()
    # Handle Empty Trajectory
    if traj.get_number_of_segments() < 1:
      robot_setpoint.get_mutable_value()[:] = self.EvalVectorInput(context, self._robot_state_input_port.get_index()).get_value()[:]
    else: 
      n_pos = robot_setpoint.get_mutable_value()
      robot_setpoint.get_mutable_value()[:robot_setpoint.size()//2] = traj.value(time).flatten()
      robot_setpoint.get_mutable_value()[robot_setpoint.size()//2:] = traj.derivative().value(time).flatten()

  def poll_lcm_target(self, context, state_next):
        self.ValidateContext(context)
        # Evaluate the input ports to obtain the current command
        target_kinematics = self._target_kinematics_input_port.Eval(context)
        target_timestamp = context.get_mutable_discrete_state(self._prev_target_timestamp_idx)
        # Do Nothing if command didn't change
        if target_timestamp[0] >= target_kinematics.utime:
          return EventStatus.DidNothing()

        # If command did change, re-calculate the target trajectory
        knots = np.array([context.get_time(), context.get_time() + self._traj_time_len])
        cur_state = self.EvalVectorInput(context, self._robot_state_input_port.get_index()).get_value()

        # Generate target state
        target_state = np.copy(cur_state)
        # Use current state as target state in trajectory at first loop
        if target_timestamp[0] >= 0:
          for finger_idx, state_idx in enumerate(finger_idx_from_body_name(self._plant, self._robot_id, self._body_names)):
            pos_idx = 3*state_idx
            vel_idx = len(target_state) // 2 + pos_idx
            target_state[pos_idx:pos_idx+3] = target_kinematics.targetPos[3*finger_idx:3*finger_idx+3]
            target_state[vel_idx:vel_idx+3] = target_kinematics.targetVel[3*finger_idx:3*finger_idx+3]
          # If relative to current state:
          if not target_kinematics.isAbsoluteTargetPos:
            target_state[:len(target_state)//2] += cur_state[:len(target_state)//2]

        # Generate Trajectory
        samples = np.vstack([cur_state[:len(target_state)//2], target_state[:len(target_state)//2]]).T
        samples_dot = np.vstack([cur_state[len(target_state)//2:], target_state[len(target_state)//2:]]).T
        traj = PiecewisePolynomial.CubicHermite(knots, samples, samples_dot)

        # State updates
        target_timestamp[0] = target_kinematics.utime
        state_next.get_mutable_discrete_state().set_value(np.array([target_timestamp[0]]))
        state_next.get_mutable_abstract_state().get_mutable_value(self._trajectory_index).SetFrom(Value(traj))
        return EventStatus.Succeeded()


@gin.configurable(denylist=['plant'])
class DensetactIOSystem(LeafSystem):
  """Create a Drake ``LeafSystem`` which converts contact data
  to LCM Messages
  """
  def __init__(self, plant: MultibodyPlant, robot_id: ModelInstanceIndex, densetact_body_names: List[str], normal_scale: float = 1.0, friction_scale: float = 1.0):
    super().__init__()

    self._body_names = densetact_body_names
    self._plant = plant
    self._robot_id = robot_id
    self._normal_scale = normal_scale
    self._friction_scale = friction_scale

    # Create an input port for averaged contact data
    self._avg_contact_input_port = self.DeclareAbstractInputPort(
        "averaged_contact_data", Value({"force": dict(),"point": dict(),"normal": dict()})
    )
    self._robot_state_input_port = self.DeclareVectorInputPort(
      "robot_state", BasicVector(self._plant.num_multibody_states(self._robot_id))
    )

    self.DeclareAbstractOutputPort("lcmt_densetact_measurement_data",
                                   lambda: Value(lcmt_densetact_measurement_data()),
                                   self.calc_densetact_output)

  def calc_densetact_output(self, context, densetact_msg):
      self.ValidateContext(context)
      # Evaluate the input ports to obtain the averaged contact results
      robot_state = self.EvalVectorInput(context, self._robot_state_input_port.get_index())
      avg_contact = self._avg_contact_input_port.Eval(context)

      sensorized_bodies = self._body_names[:2]

      densetact_msg.get_mutable_value().numSensors = len(sensorized_bodies)
      densetact_msg.get_mutable_value().sensorData.clear()
      utime = int(context.get_time() * 1e6)

      for fingertip_idx, body_name in zip(finger_idx_from_body_name(self._plant, self._robot_id, self._body_names), self._body_names):
        #import pdb; pdb.set_trace()

        body_idx = int(self._plant.GetBodyByName(body_name).index())
        fingertip_pose_W = np.array([robot_state[fingertip_idx * 3], robot_state[fingertip_idx * 3 + 1], robot_state[fingertip_idx * 3 + 2]])
        
        R_BW = self._plant.GetFrameByName(body_name).GetFixedRotationMatrixInBodyFrame().matrix()
        
        measurement = lcmt_densetact_measurement()
        
        force_W = np.array(avg_contact["force"][body_idx])
        point = np.array(avg_contact["point"][body_idx]) - fingertip_pose_W
        normal_W = np.array(avg_contact["normal"][body_idx])
        measurement.utime = utime
        measurement.inContact = not np.all(np.isclose(normal_W, np.zeros_like(normal_W)))
        
        #import pdb; pdb.set_trace()
        if measurement.inContact:
          R_CW = DensetactIOSystem.rotation_matrix_from_vectors(np.array([0., 0., 1.]), normal_W)
          
          # sensor1_pose_W.EvalPoseInWorld(context)
          #import pdb; pdb.set_trace()
          R_CB = R_BW.T @ R_CW
          assert not np.any(np.isnan(R_CW))
          force_C = R_CW.T.dot(force_W)
          for idx in range(3):
            for jdx in range(3):
              # Copy Rotation of body frame -> contact frame
              measurement.contactPose[idx][jdx] = R_CW[idx][jdx]
            # Copy Translation, i.e., contact point in body frame
            measurement.contactPose[idx][3] = point[idx]
          measurement.contactPose[3][3] = 1.0 # Valid affine transform
          # Force in contact frame
          measurement.scaledNormal = self._normal_scale * force_C[2]
          measurement.scaledFriction[0] = self._friction_scale * force_C[0]
          measurement.scaledFriction[1] = self._friction_scale * force_C[1]
        else:
          # Identity Transform, leave 0 forces
          for idx in range(4):
            measurement.contactPose[idx][idx] = 1.0
        # Add Measurement data
        densetact_msg.get_mutable_value().sensorData.append(measurement)

  # From https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
  @staticmethod
  def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    vec1_norm, vec2_norm = (vec1 / (np.linalg.norm(vec1))).reshape(3), (vec2 / (np.linalg.norm(vec2))).reshape(3)
    # Special case: same vector
    if np.all(np.isclose(vec1_norm, vec2_norm)):
      return np.eye(3)
    # Special case: opposite vectors (use x rotation)
    if np.all(np.isclose(vec1_norm, -vec2_norm)):
      vec1_norm += np.finfo(vec1_norm.dtype).eps * np.random.rand(3)
      vec1_norm = (vec1_norm / np.linalg.norm(vec1_norm)).reshape(3)
    cross = np.cross(vec1_norm, vec2_norm)
    dot = np.dot(vec1_norm, vec2_norm)
    norm = np.linalg.norm(cross)
    kmat = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - dot) / (norm ** 2))
    return rotation_matrix


@gin.configurable
class FingerTipIOSystem(LeafSystem):
    """Create a Drake ``LeafSystem`` which converts LCM messages
    to usable system vectors
    """

    def __init__(self, plant: MultibodyPlant, object_id: ModelInstanceIndex, fingertip_body_names: List[str]):
        super().__init__()

        # Input Validation
        self._plant = plant
        self._fingertip_body_names = fingertip_body_names
        assert len(self._fingertip_body_names) == 3 # expects 3 fingers
        self._object_nq = plant.num_positions(object_id)
        self._object_nv = plant.num_velocities(object_id)
        self._object_name = plant.GetModelInstanceName(object_id)
        self._object_state_names = plant.GetStateNames(object_id)

        # System state input ports
        self._body_poses_port = self.DeclareAbstractInputPort(
          "plant_body_poses", plant.get_body_poses_output_port().Allocate()
        )
        self._body_vels_port = self.DeclareAbstractInputPort(
          "plant_body_velocities", plant.get_body_spatial_velocities_output_port().Allocate()
        )
        self._object_state_input_port = self.DeclareVectorInputPort(
            "object_state", BasicVector(self._object_nq + self._object_nv)
        )

        # Ouput LCM Messages
        self.DeclareAbstractOutputPort("lcmt_fingertips_position",
                                       lambda: Value(lcmt_fingertips_position()),
                                       self.calc_fingertips_position_output)
        self.DeclareAbstractOutputPort("lcmt_object_state",
                                       lambda: Value(lcmt_object_state()),
                                       self.calc_object_output)

    def calc_fingertips_position_output(self, context, fingertips_positions_msg):
        body_poses = self._body_poses_port.Eval(context)
        body_vels = self._body_vels_port.Eval(context)
        # using the time from the context
        fingertips_positions_msg.get_mutable_value().utime = int(context.get_time() * 1e6)
        # Populate position / velocity
        for body_enum, body_name in enumerate(self._fingertip_body_names):
          body_idx = self._plant.GetBodyByName(body_name).index()
          for idx in range(3):
            fingertips_positions_msg.get_mutable_value().curPos[body_enum * 3 + idx] = body_poses[body_idx].translation()[idx]
            fingertips_positions_msg.get_mutable_value().curVel[body_enum * 3 + idx] = body_vels[body_idx].translational()[idx]
        # Set Identity Quaternion (W = 1)
        fingertips_positions_msg.get_mutable_value().curQuat[0] = 1.0
        fingertips_positions_msg.get_mutable_value().curQuat[4] = 1.0
        fingertips_positions_msg.get_mutable_value().curQuat[8] = 1.0

    def calc_object_output(self, context, object_msg):
        state = self.EvalVectorInput(context, self._object_state_input_port.get_index())
        # using the time from the context
        object_msg.get_mutable_value().utime = int(context.get_time() * 1e6)
        object_msg.get_mutable_value().object_name = self._object_name
        object_msg.get_mutable_value().num_positions = self._object_nq
        object_msg.get_mutable_value().num_velocities = self._object_nv
        # Populate position / velocity
        object_msg.get_mutable_value().position.clear()
        object_msg.get_mutable_value().position_names.clear()
        object_msg.get_mutable_value().velocity.clear()
        object_msg.get_mutable_value().velocity_names.clear()
        for idx in range(self._object_nq):
          object_msg.get_mutable_value().position.append(state.GetAtIndex(idx))
          object_msg.get_mutable_value().position_names.append(self._object_state_names[idx])
        for idx in range(self._object_nv):
          object_msg.get_mutable_value().velocity.append(state.GetAtIndex(self._object_nq + idx))
          object_msg.get_mutable_value().velocity_names.append(self._object_state_names[self._object_nq + idx])


@gin.configurable(denylist=['builder', 'plant'])
def sim_diagram_builder(
    builder: DiagramBuilder,
    plant: MultibodyPlant,
    robot_model_name: str,
    object_model_name: str,
    lcm_pub_dt: float,
    lcm_densetact_dt: float,
    lcm_channels: Dict[str, str],
    pid_gains: List[float],
):
  print("sim_diagram_builder called")
  lcm = builder.AddSystem(LcmInterfaceSystem(DrakeLcm()))
  robot_model_id = plant.GetModelInstanceByName(robot_model_name)
  object_model_id = plant.GetModelInstanceByName(object_model_name)

  # LCM Output Interface
  fingertip_pos_pub = builder.AddSystem(LcmPublisherSystem(lcm_channels["fingertips_position"], PySerializer(lcmt_fingertips_position), lcm, lcm_pub_dt))
  object_state_pub = builder.AddSystem(LcmPublisherSystem(lcm_channels["object_state"], PySerializer(lcmt_object_state), lcm, lcm_pub_dt))
  fingertip_io_system = builder.AddSystem(FingerTipIOSystem(plant, object_model_id))
  builder.Connect(
        plant.get_body_poses_output_port(), fingertip_io_system.GetInputPort("plant_body_poses")
  )
  builder.Connect(
        plant.get_body_spatial_velocities_output_port(), fingertip_io_system.GetInputPort("plant_body_velocities")
  )
  builder.Connect(
        fingertip_io_system.GetOutputPort("lcmt_fingertips_position"), fingertip_pos_pub.get_input_port()
  )
  builder.Connect(
        plant.get_state_output_port(object_model_id), fingertip_io_system.GetInputPort("object_state")
  )
  builder.Connect(
        fingertip_io_system.GetOutputPort("lcmt_object_state"), object_state_pub.get_input_port()
  )

  # Contact Force Systems
  cf_averager = builder.GetMutableSubsystemByName("averager")
  densetact_io_system = builder.AddSystem(DensetactIOSystem(plant, robot_model_id))
  densetact_pub = builder.AddSystem(LcmPublisherSystem(lcm_channels["densetact"], PySerializer(lcmt_densetact_measurement_data), lcm, lcm_densetact_dt))
  builder.Connect(
        cf_averager.get_output_port(), densetact_io_system.GetInputPort("averaged_contact_data")
  )
  builder.Connect(
        plant.get_state_output_port(robot_model_id), densetact_io_system.GetInputPort("robot_state")
  )
  builder.Connect(
        densetact_io_system.GetOutputPort("lcmt_densetact_measurement_data"), densetact_pub.get_input_port()
  )

  # PID Robot Controller
  fingertips_target_sub = builder.AddSystem(LcmSubscriberSystem(lcm_channels["fingertips_target"], PySerializer(lcmt_fingertips_target_kinematics), lcm))
  fingertips_target_to_setpoint = builder.AddSystem(SetpointFromTargetSystem(plant, robot_model_id))
  control_size = plant.get_actuation_input_port(robot_model_id).size()
  pid_controller = builder.AddSystem(PidController(
        pid_gains[0] * np.ones(control_size), pid_gains[1] * np.ones(control_size), pid_gains[2] * np.ones(control_size)
    ))
  builder.Connect(
        fingertips_target_sub.get_output_port(), fingertips_target_to_setpoint.GetInputPort("lcmt_fingertips_target_kinematics")
  )
  builder.Connect(
        plant.get_state_output_port(robot_model_id), fingertips_target_to_setpoint.GetInputPort("robot_state")
  )
  builder.Connect(
        plant.get_state_output_port(robot_model_id), pid_controller.get_input_port_estimated_state()
  )
  builder.Connect(
        fingertips_target_to_setpoint.get_output_port(), pid_controller.get_input_port_desired_state()
  )
  builder.Connect(
        pid_controller.get_output_port_control(), plant.get_actuation_input_port(robot_model_id)
  )


## Main Function
@gin.configurable
def main(
  init_sim_state: Optional[Dict[str, List[float]]] = None,
  sim_rate: float = 1.0
):
    """Main function for simulation"""
    print("Simulation With Object and Robot")

    plant_diagram = MultibodyPlantDiagram()
    plant_context = plant_diagram.plant.GetMyMutableContextFromRoot(plant_diagram.sim.get_mutable_context())

    # Generate plant diagram for debugging
    plant_diagram.diagram.set_name("dairpll_simplesim")
    plt.figure(figsize=(11,8.5), dpi=300)
    plot_system_graphviz(plant_diagram.diagram)
    plt.savefig(str(Path.home() / "Desktop" / "dairpll_simplesim.png"))
    plt.close()

    # Set Initial State
    if init_sim_state is not None:
      for model_name, model_state in init_sim_state.items():
        model_id = plant_diagram.plant.GetModelInstanceByName(model_name)
        plant_diagram.plant.SetPositionsAndVelocities(plant_context, model_id, np.array(model_state))

    # Run Simulator
    print("Running Sim... Press Ctrl-C to Stop")
    plant_diagram.sim.set_target_realtime_rate(sim_rate)
    plant_diagram.sim.set_publish_every_time_step(False)
    try:
      plant_diagram.sim.AdvanceTo(float('inf'))
    except KeyboardInterrupt:
        print("\nClosing...")
        sys.exit(0)

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