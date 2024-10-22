#!/usr/bin/env python3

import os
import sys
from typing import Dict, List

import gin
import gin.torch
import git
import numpy as np
import torch
from torch import Tensor
from tensordict.tensordict import TensorDict

from dair_pll import file_utils, drake_controllers
from dair_pll.drake_system import DrakeSystem, carry_dict_create


# Repository directory (default for file operations)
REPO_DIR = os.path.normpath(
    git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")
)
DEFAULT_CONFIG = "default.gin"

# Create Initial State
@gin.configurable(denylist=['system'])
def sim_initial_state(
    system: DrakeSystem, 
    state: Dict[str, List[float]]
) -> Tensor:
    tdict = TensorDict({}, batch_size = (1,))
    for key, val in state.items():
        tdict[key + "_state"] = torch.tensor(val).reshape(1, -1)
    return system.construct_state_tensor(tdict) # Does input validation already


# Main Function
@gin.configurable
def main(
    storage_folder_name: str = "storage",
    run_name: str = "default_run",
):

    print("ContactNets With Sparse Tactile Sensing")
    storage_name = os.path.join(REPO_DIR, "results", storage_folder_name)
    print(f"Storing data at    {file_utils.data_dir(storage_name)}")
    print(f"Storing results at {file_utils.run_dir(storage_name, run_name)}")

    # Load True URDFs into Drake Base System
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

    ## TODO: Make trajectory class for different parameterizations
    ## TODO: Make above variable length

    # Start Input Loop
    def print_help():
        print("\nUsage:\n" \
            "b - breakpoint()\n" \
            "c - Collect Sim Data\n" \
            "h - Print Help\n" \
            "u - Update PID Ref\n" \
            "q - Quit\n")
    print_help()
    command_char = ' '
    while command_char != 'q':
        command_char = input('Command $ ').split(" ")[0]

        if command_char == 'h':
            print_help()

        elif command_char == 'b':
            breakpoint()

        elif command_char == 'c':
            seconds = float(input("How long (s)? "))
            state, data = base_system.simulate(initial_state, carry_dict, int(seconds/base_system.dt))
            data["state"] = state
            if sim_trajectory is None:
                sim_trajectory = torch.clone(data)
            else:
                # Adjust time and append
                if "time" in data:
                    data["time"] += sim_trajectory["time"][-1]
                sim_trajectory = torch.cat((sim_trajectory, data))
            # Update Initial State
            initial_state = sim_trajectory[-1:, :]

        elif command_char == 'u':
            print("Enter comma-space-separated floats.\n")
            updated_ref = np.array([f for f in map(float, input('New State: ').split(", "))])
            drake_controllers.update_pid_reference(base_system, updated_ref)

        elif command_char != 'q':
            print("Warning: Unrecognized command.\n")
    
    # Collect more Data: ask for amount of time to advance and knot points;
    ## Simulate Trajectory from previous X0
    ## Store new X0 and Trajectory
    # Visualize Data

    # Quit



if __name__ == "__main__":
    config_file = DEFAULT_CONFIG
    if len(sys.argv) < 2:
        print(f"Warning: Using default config file ({DEFAULT_CONFIG})")
    else:
        config_file = sys.argv[1]
    
    # Parse config file and start
    gin.parse_config_file(os.path.join(REPO_DIR, "config", config_file))
    main()
