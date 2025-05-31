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

from enum import Enum
import os
import pdb
import random
import signal
import sys
import time
from typing import cast, Any, Dict, List, Optional, Tuple, Type

import gin
import gin.torch.external_configurables
import git
import lcm
import numpy as np
from pydrake.geometry import StartMeshcat, Rgba, Shape
from pydrake.geometry import HalfSpace as DrakeHalfSpace  # type: ignore
from scipy.spatial.transform import Rotation as R
from tensordict import TensorDictBase, TensorDict
import torch
from torch.distributions.normal import Normal
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch import Tensor

from dair_pll import file_utils
from dair_pll.drake_system import DrakeSystem
from dair_pll.dataset_management import TrajectorySet
from dair_pll.gui_utils import PLLMeshcatVisualizer
from dair_pll.multibody_tactile_learnable_system import MultibodyLearnableTactileSystem
from dair_pll.lcmtypes.dairlib import (
    lcmt_fingertips_position,
    lcmt_object_state,
    lcmt_densetact_measurement_data,
    lcmt_fingertips_target_kinematics,
)
from dair_pll.tensor_utils import pbmm
from dair_pll.hack_utils import finger_idx_from_body_name

# Repository directory (default for file operations)
REPO_DIR = os.path.normpath(
    git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")
)
DEFAULT_CONFIG = "active_exploration.gin"


## Training Function
def train_epoch(
    data: DataLoader,
    system: MultibodyLearnableSystemWithTrajectory,
    optimizer: Optional[Optimizer] = None,
) -> Tensor:
    """Train learned model for a single epoch.  Takes gradient steps in the
    learned parameters if ``optimizer`` is provided.

    Args:
        data: Training dataset.
        system: System to be trained.
        optimizer: Optimizer which trains system.

    Returns:
        Scalar average training loss observed during epoch.
    """
    losses = []
    loss_elements = {}
    for xy_i in data:
        x_past: Tensor = xy_i[0]
        x_plus: Tensor = xy_i[1]

        if optimizer is not None:
            optimizer.zero_grad()

        loss = system.contactnets_loss(**get_loss_args(x_past, x_plus, system)).mean()
        losses.append(loss.clone().detach())

        for key, val in system.loss_cache.items():
            if key not in loss_elements:
                loss_elements[key] = []
            loss_elements[key].append(val)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

    # Compute Epoch Average
    avg_loss = cast(Tensor, sum(losses) / len(losses))
    loss_elements_ret = {}
    for key, val in loss_elements.items():
        loss_elements_ret[key] = cast(Tensor, sum(val) / len(val))
    return avg_loss, loss_elements_ret


### Visualization
def get_true_geometry() -> Shape:
    """Get True Geometry from configured base system"""
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


def signal_handler(sig, frame):
    """Handle SIGINT"""
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
            safe_state[2] = safe_trifinger_height
            safe_state[3:6] = (
                new_trajectory["finger_1"]["position"][-1].cpu().clone().numpy()
            )
            safe_state[5] = safe_trifinger_height
            trifinger_lcm.execute_trajectory(safe_state, no_data=True)

            # Add data to dataset
            add_trajectory = TensorDict({}, batch_size=new_trajectory.batch_size)
            add_trajectory["robot_state"] = extract_robot_trajectory(
                learned_system, new_trajectory, robot_model_name
            )
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
            add_trajectory[object_model_name + "_groundtruth"] = new_trajectory[
                object_model_name
            ]["position"]
            data_trajectories.add_trajectories(
                [add_trajectory.clone().detach()],
                torch.tensor([len(data_trajectories.trajectories)], dtype=torch.int),
            )

            learned_system.add_trajectories(
                traj_lens=[len(add_trajectory["time"])],
            )

            # Re-init visualizer
            gui_vis.update()

            # Re-init optimizer and data-loader
            batch_size = (
                len(data_trajectories.slices)
                if data_trajectories.slices.config.batch_size == -1
                else data_trajectories.slices.config.batch_size
            )
            traj_dataloader = DataLoader(
                data_trajectories.slices,
                batch_size=batch_size,
                shuffle=data_trajectories.slices.config.shuffle,
                generator=torch.Generator(device=torch.get_default_device()),
            )
            optimizer = optimizer_cls(learned_system.parameters())

        elif command_char == "s":
            try:
                action_idx = int(input("Which Action (<0 == random)? "))
            except ValueError:
                print("Cancelling...")
                continue
            if action_idx < 0:
                action_idx = None
                print(f"Sampling random action: {temp}")

            selected_action = sample_action(index=action_idx)

        elif command_char == "v":
            print("Visualizing entire trajectory.")
            gui_vis.sweep()
            print("Done!")

        elif command_char == "t":
            if traj_dataloader is None or len(traj_dataloader) == 0:
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
                train_loss, loss_data = train_epoch(
                    traj_dataloader, learned_system, optimizer
                )
                total_epochs += 1
                print(
                    total_epochs,
                    f"Loss (J): {train_loss:.3e};",
                    f"Pred (Nm): {loss_data['mean_pred_Nm']:.3e};",
                    f"Pred (<m=rad>/s): {loss_data['mean_q_pred_mps']:.3e};",
                    f"Comp (Nm): {loss_data['mean_comp_Nm']:.3e};",
                    f"Pen (m): {loss_data['mean_pen_m']:.3e};",
                    f"Diss (J/s): {loss_data['mean_diss_Jps']:.3e};",
                    f"Dev (N): {loss_data['mean_dev_N']:.3e};",
                    f"Norm (cosine): {loss_data['mean_norm_cosine']:.3e};",
                )
                gui_vis.update()
                train_losses.append(train_loss)
                train_loss_data.append(loss_data)
                learned_summaries.append(learned_system.summary({}))
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
