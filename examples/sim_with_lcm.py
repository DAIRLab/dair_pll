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
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import ModelInstanceIndex
from pydrake.systems.drawing import plot_system_graphviz
from pydrake.systems.framework import BasicVector, DiagramBuilder, LeafSystem
from pydrake.systems.lcm import LcmInterfaceSystem, LcmPublisherSystem, PySerializer

from dair_pll.drake_utils import MultibodyPlantDiagram
from dair_pll.lcmtypes.dairlib import lcmt_fingertips_position
from dair_pll import file_utils

# Repository directory (default for file operations)
REPO_DIR = os.path.normpath(
    git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")
)
DEFAULT_CONFIG = "sim_with_lcm.gin"


## Diagram Builder

class FingerTipIOSystem(LeafSystem):
    """Create a Drake ``LeafSystem`` which converts LCM messages
    to usable system vectors
    """

    def __init__(self, plant: MultibodyPlant, model_id: ModelInstanceIndex):
        super().__init__()
        assert plant.num_positions(model_id) == plant.num_velocities(model_id)
        self._model_nx = plant.num_multibody_states(model_id)
        assert self._model_nx <= 18 # expects 3D * 3 fingers, nq=nv=9; can have less

        # Create an input port for the current state of the system.
        self._robot_state_input_port = self.DeclareVectorInputPort(
            "robot_state", BasicVector(self._model_nx)
        )

        # Ouput LCM Message
        self.DeclareAbstractOutputPort("lcmt_fingertips_position",
                                       lambda: Value(lcmt_fingertips_position()),
                                       self.calc_fingertips_position_output)

    def calc_fingertips_position_output(self, context, fingertips_positions_msg):
        state = self.EvalVectorInput(context, self._robot_state_input_port.get_index())
        # using the time from the context
        fingertips_positions_msg.get_mutable_value().utime = int(context.get_time() * 1e6)
        # Populate position / velocity
        for idx in range(self._model_nx // 2):
          fingertips_positions_msg.get_mutable_value().curPos[idx] = state.GetAtIndex(idx)
          fingertips_positions_msg.get_mutable_value().curVel[idx] = state.GetAtIndex((self._model_nx // 2) + idx)


@gin.configurable(denylist=['builder', 'plant'])
def sim_diagram_builder(
    builder: DiagramBuilder,
    plant: MultibodyPlant,
    robot_model_name: str,
    lcm_pub_dt: float,
    lcm_channels: Dict[str, str],
):
  print("sim_diagram_builder called")
  lcm = builder.AddSystem(LcmInterfaceSystem(DrakeLcm()))
  robot_model_id = plant.GetModelInstanceByName(robot_model_name)

  # LCM Output Interface
  fingertip_pos_pub = builder.AddSystem(LcmPublisherSystem(lcm_channels["fingertips_position"], PySerializer(lcmt_fingertips_position), lcm, lcm_pub_dt))
  fingertip_io_system = builder.AddSystem(FingerTipIOSystem(plant, robot_model_id))
  builder.Connect(
        plant.get_state_output_port(robot_model_id), fingertip_io_system.get_input_port()
  )
  builder.Connect(
        fingertip_io_system.get_output_port(), fingertip_pos_pub.get_input_port()
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