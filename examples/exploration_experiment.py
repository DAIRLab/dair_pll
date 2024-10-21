#!/usr/bin/env python3

import os
import sys

import gin
import gin.torch
import git


# Repository directory (default for file operations)
REPO_DIR = os.path.normpath(
    git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")
)
DEFAULT_CONFIG = "default.gin"

@gin.configurable
def main(
    storage_folder_name: str = "storage",
    run_name: str = "default_run",
    system: str = CUBE_SYSTEM,
    source: str = SIM_SOURCE,
    structured: bool = True,
    contactnets: bool = True,
    geometry: str = BOX_TYPE,
    regenerate: bool = False,
    dataset_size: int = 1,
    learnable_params: int = 0b11000,  # Default no inertia
    true_sys: bool = True,
    wandb_project: str = WANDB_DEFAULT_PROJECT,
    w_pred: float = 1e0,
    w_comp: float = 1e0,
    w_diss: float = 1e0,
    w_pen: float = 2e1,
    w_res: float = 1e0,
    w_res_w: float = 1e0,
    w_dev: float = 2e1,
    do_residual: bool = False,
    g_frac: float = 1.0,
):

    print("ContactNets With Sparse Tactile Sensing")
    print(f"Storing data at    {file_utils.data_dir(storage_name)}")
    print(f"Storing results at {file_utils.run_dir(storage_name, run_name)}")



if __name__ == "__main__":
    config_file = DEFAULT_CONFIG
    if len(sys.argv) < 2:
        print(f"Warning: Using default config file ({DEFAULT_CONFIG})")
    else:
        config_file = sys.argv[1]
    
    # Parse config file and start
    gin.parse_config_file(os.path.join(REPO_DIR, "config", config_file)
    main()
