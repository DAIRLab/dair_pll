import gin
import numpy as np
from typing import List, Tuple
from enum import Enum

import sys
import os
import git

#gin.parse_config_file("config/sim_experiment.gin")

# DEFAULT_CONFIG = "rss_experiment.gin"
# REPO_DIR = os.path.normpath(
#     git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")
# )
# config_file = DEFAULT_CONFIG
# if len(sys.argv) < 2:
#     print(f"Warning: Using default config file ({DEFAULT_CONFIG})")
# else:
#     config_file = sys.argv[1]

# # Parse config file and start
# gin.parse_config_file(os.path.join(REPO_DIR, "config", config_file))

class ActionLibrary(Enum):
    NONE = 0
    XPINCH = 1
    YPINCH = 2
    ZPINCH = 3
    XSINGLE = 4
    YSINGLE = 5
    ZSINGLE = 6
    CORNERSINGLE = 7
    EDGESINGLE = 8

@gin.configurable
def sample_action(
    workspace_xy_center: Tuple[float, float],
    workspace_z_rot: float,
    workspace_radius: float,
    sphere_radius: float,
    fixed_240_W: List[float],
    library: ActionLibrary = ActionLibrary.NONE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a straight line action
    Params:
    workspace_xy_center: offset the workspace from world origin
    workspaxe_z_rot: world rotation so finger_0 is towards +X axis
    workspace_radius: start will be along edge of radius above ground
    sphere_radius: radius of robot fingertip
    fixed_240: 3d position
    library: int for fixed set of action
    """

    assert workspace_radius > 0.0
    assert 0.0 < sphere_radius < workspace_radius
    assert len(fixed_240_W) == 3
    fixed_240_traj = np.array(fixed_240_W)
    rng = np.random.default_rng()

    # Start in workspace frame
    def sample_finger(flip_x: bool = False, library = library):
        flip_factor = -1.0 if flip_x else 1.0
        start_polar = rng.uniform(0.0, np.pi / 2.0)
        start_azimuth = rng.uniform(-np.pi / 2.0, np.pi / 2.0)
        if library in (ActionLibrary.XPINCH, ActionLibrary.XSINGLE):
            start_polar = np.pi / 2.0
            start_azimuth = 0.
        elif library in (ActionLibrary.ZPINCH, ActionLibrary.ZSINGLE):
            start_polar = 0.
            start_azimuth = 0.
        elif library in (ActionLibrary.YPINCH, ActionLibrary.YSINGLE):
            start_polar = np.pi / 2.0
            start_azimuth = flip_factor * (np.pi / 2.0)
        elif library in (ActionLibrary.CORNERSINGLE,):
            start_polar = np.pi / 5.0
            start_azimuth = np.pi / 2.5
        elif library in (ActionLibrary.EDGESINGLE,):
            start_polar = np.pi / 7.0
            start_azimuth = 0.
        if flip_x and (library in (ActionLibrary.XSINGLE, ActionLibrary.YSINGLE, ActionLibrary.ZSINGLE, ActionLibrary.CORNERSINGLE, ActionLibrary.EDGESINGLE)):
            start_polar = 0.
            start_azimuth = 0.
        start_S = (workspace_radius - sphere_radius) * np.array(
            [
                (np.sin(start_polar) * np.cos(start_azimuth)),
                np.sin(start_polar) * np.sin(start_azimuth),
                np.cos(start_polar),
            ]
        )
        start_S[0] += sphere_radius
        start_S[2] += sphere_radius
        start_S[0] *= flip_factor
        max_radius = workspace_radius - sphere_radius
        end_radius = rng.uniform(0.0, max_radius)
        end_angle = rng.uniform(0.0, np.pi)
        if not (library is ActionLibrary.NONE):
            end_radius = 0.
            end_angle = 0.
        if flip_x and (library in (ActionLibrary.XSINGLE, ActionLibrary.YSINGLE, ActionLibrary.ZSINGLE, ActionLibrary.CORNERSINGLE)):
            end_radius = max_radius
            end_angle = np.pi / 2.0
        end_S = np.array(
            [
                flip_factor * sphere_radius,
                end_radius * np.cos(end_angle),
                sphere_radius + end_radius * np.sin(end_angle),
            ]
        )
        return (start_S, end_S)

    finger_0_traj = sample_finger(False)
    finger_120_traj = sample_finger(True)

    ret = (np.zeros(18), np.zeros(18))
    z_rot = np.array(
        [
            [np.cos(workspace_z_rot), -np.sin(workspace_z_rot), 0.0],
            [np.sin(workspace_z_rot), np.cos(workspace_z_rot), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    xy_trans = np.array([workspace_xy_center[0], workspace_xy_center[1], 0.0])
    for idx in [0, 1]:
        ret[idx][:3] = z_rot @ finger_0_traj[idx].T + xy_trans
        ret[idx][3:6] = z_rot @ finger_120_traj[idx].T + xy_trans  
        ret[idx][6:9] = fixed_240_traj[:]
    return ret

    # def load_gin():
    #     config_file = DEFAULT_CONFIG
    #     if len(sys.argv) < 2:
    #         print(f"Warning: Using default config file ({DEFAULT_CONFIG})")
    #     else:
    #         config_file = sys.argv[1]

    #     # Parse config file and start
    #     gin.parse_config_file(os.path.join(REPO_DIR, "config", config_file))