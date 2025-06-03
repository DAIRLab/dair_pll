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

# pylint: disable=invalid-name,too-many-statements,too-many-locals,too-many-branches

import os
import pdb
import signal
import sys
import time
from typing import Type

import gin
import gin.torch.external_configurables
import git
import numpy as np
from pydrake.geometry import Shape
from pydrake.geometry import HalfSpace as DrakeHalfSpace  # type: ignore
from tensordict import TensorDict
import torch

from dair_pll import file_utils
from dair_pll.drake_system import DrakeSystem
from dair_pll.dataset_management import TrajectorySet
from dair_pll.gui_utils import PLLMeshcatVisualizer
from dair_pll.hack_utils import extract_robot_trajectory
from dair_pll.multibody_tactile_learnable_system import MultibodyLearnableTactileSystem
from dair_pll.trifinger_utils import (
    TrifingerLCMService,
    sample_action,
    interpolate_sampled_action,
)

# Repository directory (default for file operations)
REPO_DIR = os.path.normpath(
    git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")
)
DEFAULT_CONFIG = "active_exploration.gin"


## Training Function
# TODO: Add Training Function


### Visualization
def get_true_geometry() -> Shape:
    """Get True Geometry from configured base system"""
    # Pylint doesn't know about gin
    # pylint: disable=no-value-for-parameter
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


def signal_handler(_sig, _frame):
    """Handle SIGINT"""
    # pylint: disable=global-statement
    global signal_pressed
    signal_pressed = True


## Main Function
@gin.configurable
def main(
    storage_folder_name: str = "storage_active",
    run_name: str = "default_run",
    optimizer_cls: Type = torch.optim.SGD,
):
    """Main function for online learning loop"""
    # pylint: disable=global-statement
    global signal_pressed
    signal.signal(signal.SIGINT, signal_handler)
    # torch.autograd.set_detect_anomaly(True) ## NOTE: doesn't work with vmap
    # Debug: Remove scientific notation for numpy printing
    np.set_printoptions(suppress=True)
    torch.set_default_device("cuda")

    # Create run directory
    print("Active Tactile Exploration")
    storage_name = os.path.join(REPO_DIR, "results", storage_folder_name)
    print(f"Storing data and results at {file_utils.run_dir(storage_name, run_name)}")

    # Create learnable system
    print("Loading Learned System...")
    # Pylint doesn't know about gin
    # pylint: disable=no-value-for-parameter
    learned_system = MultibodyLearnableTactileSystem()

    # Create Dataset
    data_trajectories = TrajectorySet()

    # GUI Visualization
    gui_vis = PLLMeshcatVisualizer(
        system=learned_system, data=data_trajectories, true_geom=get_true_geometry()
    )

    # Initialize LCM
    trifinger_lcm = TrifingerLCMService()
    print("Sample Initial Random Action...")
    selected_action = sample_action()
    new_trajectory = None

    # Initialize Optimizer and Data config
    optimizer = optimizer_cls(learned_system.parameters())
    total_epochs = 0

    # Start Input Loop
    def print_help():
        print(
            "\nUsage:\n"
            "e - Execute selected action + collect data\n"
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
            safe_state[2] = trifinger_lcm.safe_height
            safe_state[3:6] = (
                new_trajectory["finger_1"]["position"][-1].cpu().clone().numpy()
            )
            safe_state[5] = trifinger_lcm.safe_height
            trifinger_lcm.execute_trajectory(safe_state, no_data=True)

            # Add data to dataset
            add_trajectory = TensorDict({}, batch_size=new_trajectory.batch_size)
            add_trajectory[learned_system.controlled_model_names[0] + "_state"] = (
                extract_robot_trajectory(new_trajectory, learned_system, trifinger_lcm)
            )
            add_trajectory[learned_system.controlled_model_names[0] + "_desired"] = (
                extract_robot_trajectory(
                    interpolate_sampled_action(
                        data=torch.tensor(np.array(selected_action)),
                        trifinger=trifinger_lcm,
                        traj_len_s=(
                            new_trajectory["time"][-1] - new_trajectory["time"][0]
                        ),
                        traj_n_steps=len(new_trajectory),
                    )[0],
                    learned_system,
                    trifinger_lcm,
                )
            )
            # TensorDict requires keys() call
            # pylint: disable=consider-using-dict-items
            for finger_name in new_trajectory.keys():
                try:
                    add_trajectory["contact_forces", finger_name] = new_trajectory[
                        finger_name
                    ]["contact_force_W"]
                    add_trajectory["contact_normals", finger_name] = new_trajectory[
                        finger_name
                    ]["contact_normal_W"]
                except (IndexError, KeyError):  # e.g. object, time
                    continue
            add_trajectory["time"] = new_trajectory["time"]
            add_trajectory[learned_system.learned_model_names[0] + "_groundtruth"] = (
                new_trajectory[learned_system.learned_model_names[0]]["position"]
            )
            data_trajectories.add_trajectories(
                [add_trajectory.clone().detach()],
                torch.tensor([len(data_trajectories.trajectories)], dtype=torch.int),
            )

            learned_system.add_learnable_trajectories(
                traj_lens=[len(add_trajectory["time"])],
            )

            # Re-init visualizer
            print("Getting current trajectory and visualizing")
            temp = learned_system(
                ctrl_desired=data_trajectories.get_full_trajectory(
                    key=learned_system.controlled_model_names[0] + "_desired"
                ),
                timestamps=data_trajectories.get_full_trajectory(key="time"),
                ctrl_actual=data_trajectories.get_full_trajectory(
                    key=learned_system.controlled_model_names[0] + "_state"
                ),
            )
            gui_vis.learned_plant_traj = temp[0]
            gui_vis.update()

            # Re-init optimizer and data-loader
            optimizer = optimizer_cls(learned_system.parameters())

        elif command_char == "s":
            try:
                action_idx = int(input("Which Action (<0 == random)? "))
            except ValueError:
                print("Cancelling...")
                continue
            if action_idx < 0:
                action_idx = None
                print("Sampling random action...")

            selected_action = sample_action(index=action_idx)

        elif command_char == "v":
            print("Visualizing entire trajectory.")
            gui_vis.sweep()
            print("Done!")

        elif command_char == "t":
            if len(data_trajectories.trajectories) == 0:
                print("Cannot train without data.\n")
                continue

            try:
                epochs = int(input("How many epochs? "))
            except ValueError:
                print("Cancelling...")
                continue
            print("Training...")

            start_time = time.time()
            for idx in range(epochs):
                # TODO: Add Training Function
                if signal_pressed:
                    signal_pressed = False
                    print("Training cancelled...")
                    epochs = idx + 1
                    break

            print(
                f"Finished training {epochs} epochs in {time.time()-start_time} seconds!"
            )

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
