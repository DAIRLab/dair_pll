#!/usr/bin/env python3
"""
Run a simulated online learning experiment
"""

import os
import sys
from typing import cast, Any, Dict, List, Type, Optional

import gin
import gin.torch.external_configurables
import git
import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tensordict.tensordict import TensorDict

# tensor_utils required for TensorDict's collate_fn
from dair_pll import file_utils, drake_controllers, tensor_utils
from dair_pll.data_config import TrajectorySliceConfig
from dair_pll.dataset_management import TrajectorySliceDataset
from dair_pll.drake_system import DrakeSystem, carry_dict_create
from dair_pll.drake_utils import get_body_names_in_model_instance
from dair_pll.multibody_learnable_system import MultibodyLearnableSystemWithTrajectory


# Repository directory (default for file operations)
REPO_DIR = os.path.normpath(
    git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")
)
DEFAULT_CONFIG = "default.gin"


# Create Initial State
@gin.configurable(denylist=["system"])
def sim_initial_state(system: DrakeSystem, state: Dict[str, List[float]]) -> Tensor:
    """Create initial state for simulator"""
    tdict = TensorDict({}, batch_size=(1,))
    for key, val in state.items():
        tdict[key + "_state"] = torch.tensor(val).reshape(1, -1)
    return system.construct_state_tensor(tdict)  # Does input validation already


## Training Functions
@gin.configurable
def get_loss_args(
    x_past: Tensor,
    x_future: Tensor,
    system: DrakeSystem,
    object_body_name: str,
) -> Dict[str, Any]:
    """Convert dataloader trajectory slices into arguments for contactnets loss"""

    # Get last time of past and first of future
    # Remove extraneous dimensions
    past = x_past[..., -1, :].squeeze(-1)
    plus = x_future[..., 0, :].squeeze(-1)

    # Construct State
    x_past = system.construct_state_tensor(past)
    x_plus = system.construct_state_tensor(plus)

    # Actuation
    control = past["net_actuation"]
    if len(control.shape) == 1:
        control = control.reshape(control.shape[0], 1)

    # Construct measured contact forces on obj_b from obj_a
    # Defined as Dict: {(str(obj_a_name), str(obj_b_name)) -> R^3 force on obj_b in World Frame}
    # TODO: specify incoming data reference frame
    contact_forces = {}
    if "contact_forces" in past.keys():
        for key in past["contact_forces"].keys():
            contact_forces[(object_body_name, key)] = past["contact_forces"][key]

    return {
        "x": x_past,
        "u": control,
        "x_plus": x_plus,
        "contact_forces": contact_forces,
    }


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
    for xy_i in data:
        x_past: Tensor = xy_i[0]
        x_plus: Tensor = xy_i[1]

        if optimizer is not None:
            optimizer.zero_grad()
        # pylint: disable=E1120
        # Expect gin to handle missing arguments
        loss = system.contactnets_loss(**get_loss_args(x_past, x_plus, system)).mean()
        losses.append(loss.clone().detach())

        if optimizer is not None:
            loss.backward()
            optimizer.step()

    avg_loss = cast(Tensor, sum(losses) / len(losses))
    return avg_loss


## Main Function
@gin.configurable
def main(
    storage_folder_name: str = "storage",
    run_name: str = "default_run",
    optimizer_cls: Type = torch.optim.SGD,
):
    """Main function for online learning loop"""
    # pylint: disable=R0915, R0912, R0914
    # Expect main function to have a lot of statements and branches and variables
    # pylint: disable=E1120
    # Expect gin to handle missing arguments

    print("ContactNets With Sparse Tactile Sensing")
    storage_name = os.path.join(REPO_DIR, "results", storage_folder_name)
    print(f"Storing data at    {file_utils.data_dir(storage_name)}")
    print(f"Storing results at {file_utils.run_dir(storage_name, run_name)}")

    # Load True URDFs into Drake Base System
    print("Loading Base System...")
    base_system = DrakeSystem()
    # Constructs initial state vector (if not using prev. trajectory)
    initial_state = sim_initial_state(base_system)
    # Determines what data is recorded
    carry_dict = carry_dict_create(base_system)
    # Set system to initial state
    base_system.preprocess_initial_condition(initial_state, carry_dict)

    # Start with None current sim_trajectory
    sim_trajectory = None

    # Load False URDFs into Learned System
    print("Loading Learned System...")
    learned_system = MultibodyLearnableSystemWithTrajectory(
        output_urdfs_dir=file_utils.get_learned_urdf_dir(storage_name, run_name)
    )
    learned_summary = learned_system.summary({})
    # Initialize Optimizer and Data config
    optimizer = optimizer_cls(learned_system.parameters())
    data_config = TrajectorySliceConfig()
    traj_dataloader = None
    total_epochs = 0

    # Start Input Loop
    def print_help():
        print(
            "\nUsage:\n"
            "b - breakpoint()\n"
            "c - Collect Sim Data\n"
            "h - Print Help\n"
            "t - Train\n"
            "u - Update PID Ref\n"
            "q - Quit\n"
        )

    print_help()
    command_char = " "
    while command_char != "q":
        command_char = input("Command $ ").split(" ")[0]

        if command_char == "h":
            print_help()

        elif command_char == "b":
            breakpoint()

        elif command_char == "c":
            seconds = float(input("How long (s)? "))
            if seconds <= 0.0:
                continue
            state, data = base_system.simulate(
                initial_state, carry_dict, int(seconds / base_system.dt)
            )
            data["state"] = state.unsqueeze(-2)
            if sim_trajectory is None:
                sim_trajectory = torch.clone(data)
            else:
                # Adjust time and append
                if "time" in data:
                    data["time"] += sim_trajectory["time"][-1]
                sim_trajectory = torch.cat((sim_trajectory, data))

            # Extend Learnable Trajectory
            learned_system.extend_traj_to(float(sim_trajectory[-1]["time"]))

            # Re-init optimizer and data-loader
            traj_dataset = TrajectorySliceDataset(data_config)
            traj_dataset.add_slices_from_trajectory(sim_trajectory)
            batch_size = (
                len(traj_dataset)
                if data_config.batch_size == -1
                else data_config.batch_size
            )
            traj_dataloader = DataLoader(
                traj_dataset,
                batch_size=batch_size,
                shuffle=data_config.shuffle,
                generator=torch.Generator(device=torch.get_default_device()),
            )
            optimizer = optimizer_cls(learned_system.parameters())

            # Update Initial State
            initial_state = state[-1:, :]

        elif command_char == "t":
            if sim_trajectory is None or traj_dataloader is None:
                print("Cannot train without sim data.\n")
                continue

            epochs = int(input("How many epochs? "))
            print("Training...")

            for _ in range(epochs):
                train_loss = train_epoch(traj_dataloader, learned_system, optimizer)
                total_epochs += 1
                print(total_epochs, train_loss)

            learned_summary = learned_system.summary({})

            print("Training done!")

        elif command_char == "u":
            print("Enter comma-space-separated floats.\n")
            updated_ref = np.array(list(map(float, input("New State: ").split(", "))))
            try:
                drake_controllers.update_pid_reference(base_system, updated_ref)
            except AssertionError:
                print("Incorrect state size.")

        elif command_char != "q":
            print("Warning: Unrecognized command.\n")

    # Collect more Data: ask for amount of time to advance and knot points;
    ## Simulate Trajectory from previous X0
    ## Store new X0 and Trajectory
    # Visualize Data

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
