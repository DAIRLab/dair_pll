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
import signal
import sys
import time
from typing import cast, Any, Dict, List, Optional, Tuple, Type


import dair_pll.vis_utils as vis_utils


import gin
import gin.torch.external_configurables
import git
import lcm
import numpy as np
from pydrake.all import StartMeshcat, Rgba, Shape
from pydrake.geometry import HalfSpace as DrakeHalfSpace  # type: ignore
from scipy.spatial.transform import Rotation as R

from tensordict import TensorDictBase, TensorDict
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch import Tensor

from trifinger_lcm_service import TrifingerLCMService
from action_library import sample_action, ActionLibrary


from rss_visualization import visualize_geometries, get_true_geometry

# tensor_utils required for TensorDict's collate_fn
# pylint: disable-next=unused-import
from dair_pll import tensor_utils
from dair_pll import file_utils
from dair_pll.drake_system import DrakeSystem
from dair_pll.dataset_management import TrajectorySet
from dair_pll.gui_utils import PLLMeshcatVisualizer
from dair_pll.multibody_learnable_system import MultibodyLearnableSystemWithTrajectory
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
DEFAULT_CONFIG = "rss_experiment.gin"


def load_configs():
    config_file = DEFAULT_CONFIG
    if len(sys.argv) < 2:
        print(f"Warning: Using default config file ({DEFAULT_CONFIG})")
    else:
        config_file = sys.argv[1]
    gin.parse_config_file(os.path.join(REPO_DIR, "config", config_file))

## Training Functions
@gin.configurable
def get_loss_args(
    x_past: Tensor,
    x_future: Tensor,
    system: DrakeSystem,
    impulses: Optional[Tensor] = None,
    object_body_name: str = "cube",
) -> Dict[str, Any]:
    """Convert dataloader trajectory slices into arguments for contactnets loss"""

    # Get last time of past and first of future
    # Remove extraneous dimensions
    past = x_past[..., -1]
    plus = x_future[..., 0]

    # Construct State
    x_past = system.construct_state_tensor(past)
    x_plus = system.construct_state_tensor(plus)

    # Actuation
    n_control = system.plant_diagram.plant.num_actuated_dofs()
    control = torch.zeros(past.batch_size + (n_control,))
    if "net_actuation" in past.keys():
        control = past["net_actuation"]
        if len(control.shape) == 1:
            control = control.unsqueeze(-1)

    # Construct measured contact forces on obj_b from obj_a
    # Defined as Dict: {(str(obj_a_name), str(obj_b_name)) -> R^3 force on obj_b in World Frame}
    # TODO: specify incoming data reference frame, default World
    contact_forces = {}
    if "contact_forces" in past.keys():
        for key in past["contact_forces"].keys():
            contact_forces[(object_body_name, key)] = past["contact_forces"][key]
    contact_normals = {}
    if "contact_normals" in past.keys():
        for key in past["contact_normals"].keys():
            contact_normals[(object_body_name, key)] = past["contact_normals"][key]

    ret = {
        "x": x_past,
        "u": control,
        "x_plus": x_plus,
        "contact_forces": contact_forces,
        "contact_normals": contact_normals,
    }
    if impulses is not None:
        ret["impulses"] = impulses

    return ret

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
            optimizer.zero_grad(set_to_none=True)
        # pylint: disable=E1120
        # Expect gin to handle missing arguments
        ### Profiling
        #import cProfile, pstats, io
        #from pstats import SortKey
        #pr = cProfile.Profile()
        #pr.enable()
        loss = system.contactnets_loss(**get_loss_args(x_past, x_plus, system)).mean()
        losses.append(loss.clone().detach())

        for key, val in system.loss_cache.items():
            if key not in loss_elements:
                loss_elements[key] = []
            loss_elements[key].append(val)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        ### Profiling
        #s = io.StringIO()
        #sortby = SortKey.CUMULATIVE
        #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #ps.print_stats()
        #print(s.getvalue())
        #breakpoint()

    # Compute Epoch Average
    avg_loss = cast(Tensor, sum(losses) / len(losses))
    loss_elements_ret = {}
    for key, val in loss_elements.items():
        loss_elements_ret[key] = cast(Tensor, sum(val) / len(val))
    return avg_loss, loss_elements_ret


@gin.configurable(denylist=["system", "data"])
def extract_robot_trajectory(
    system: MultibodyLearnableSystemWithTrajectory,
    data: TensorDictBase,
    robot_model_name: str,
    fingertip_body_names: List[str],
) -> Tensor:
    """
    Params:
        data: batched TensorDict from execute_trajectory()

    Returns:
        (batch, robot_space_nx)
    """

    assert len(data.size()) >= 1
    batch_dims = data.size()
    plant = system.plant_diagram.plant
    robot_space = system.get_model_space(robot_model_name)
    ret = torch.zeros(batch_dims + (robot_space.n_x,))

    for finger_name, state_idx in zip(fingertip_body_names,
        finger_idx_from_body_name(plant, plant.GetModelInstanceByName(robot_model_name), fingertip_body_names),):
        pos_idx = 3 * state_idx
        vel_idx = robot_space.n_x // 2 + pos_idx
        ret[..., pos_idx : pos_idx + 3] = data[finger_name]["position"]
        ret[..., vel_idx : vel_idx + 3] = data[finger_name]["velocity"]

    return ret


### Signal Handling
signal_pressed = False
def signal_handler(sig, frame):
    """ Handle SIGINT"""
    global signal_pressed
    signal_pressed = True

## Main Function
@gin.configurable
def main(
    init_trifinger_state: List[float],
    safe_trifinger_height: float,
    robot_model_name: str,
    object_model_name: str = "cube",
    n_actions_optimized: int = 30,
    storage_folder_name: str = "storage_rss",
    run_name: str = "default_run",
    optimizer_cls: Type = torch.optim.SGD,
):
    """Main function for online learning loop"""
    global signal_pressed
    signal.signal(signal.SIGINT, signal_handler)
    #torch.autograd.set_detect_anomaly(True) ## NOTE: doesn't work with vmap
    torch.set_default_device("cuda")

    # Create run directory
    print("Active Tactile Exploration")
    storage_name = os.path.join(REPO_DIR, "results", storage_folder_name)
    print(f"Storing data and results at {file_utils.run_dir(storage_name, run_name)}")
    
    # Create learnable system
    print("Loading Learned System...")
    learned_system = MultibodyLearnableSystemWithTrajectory(output_urdfs_dir=
                                                file_utils.get_learned_urdf_dir(storage_name, run_name))

    learned_summaries = [learned_system.summary({})]
    train_losses = []
    train_loss_data = []

    # Create Dataset
    data_trajectories = TrajectorySet()

    # GUI Visualization
    gui_vis = PLLMeshcatVisualizer(
        system = learned_system,
        data = data_trajectories,
        true_geom = get_true_geometry()
    )

    # Initialize LCM
    # Pylint doesn't know about gin
    # pylint: disable=no-value-for-parameter
    trifinger_lcm = TrifingerLCMService(learned_system)
    print("Move to initial trifinger state")
    trifinger_lcm.execute_trajectory(np.array(init_trifinger_state), no_data=True)
    print("Sample Initial Random Action...")
    selected_action = sample_action(library=ActionLibrary.XSINGLE)#, workspace_z_rot = np.pi/4, workspace_radius = 0.1, sphere_radius=0.0175, fixed_240_W=init_trifinger_state[6:9])
    new_trajectory = None

    # Initialize Optimizer and Data config
    optimizer = optimizer_cls(learned_system.parameters())
    traj_dataloader = None
    obs_info_inv = None
    total_epochs = 0

    cube_state = None
    true_geometry = get_true_geometry() 

    # Visualization
    print("Starting Meshcat")
    vis_meshcat = StartMeshcat()

    # Start Input Loop
    def print_help():
        print(
            "\nUsage:\n"
            "a - Action Selection\n"
            "e - Execute selected action + collect data\n"
            "o - Observed info\n"
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

        elif command_char == "i":
            print("Move to initial trifinger state")
            trifinger_lcm.execute_trajectory(np.array(init_trifinger_state), no_data=True)


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
            add_trajectory = TensorDict({}, batch_size = new_trajectory.batch_size)
            add_trajectory["robot_state"] = extract_robot_trajectory(learned_system, new_trajectory, robot_model_name)
            for finger_name in new_trajectory.keys():
                try:
                    add_trajectory["contact_forces", finger_name] = new_trajectory[finger_name]["contact_force_W"]
                    add_trajectory["contact_normals", finger_name] = new_trajectory[finger_name]["contact_normal_W"]
                except (IndexError, KeyError): # e.g. object, time
                    continue
            add_trajectory["time"] = new_trajectory["time"]
            add_trajectory[object_model_name + "_groundtruth"] = new_trajectory[object_model_name]["position"]
            import pdb; pdb.set_trace()
            data_trajectories.add_trajectories(
                [add_trajectory.clone().detach()],
                torch.tensor([len(data_trajectories.trajectories)], dtype=torch.int),
            )

            # Simulate and Extend Learnable Trajectory
            # TODO: HACK Sim causes things to go flying, debug
            """
            print("Simulating init trajectory")
            with torch.no_grad():
                plant_states_dict, _, _ = learned_system.diff_simulate(
                    add_trajectory["robot_state"].unsqueeze(0), add_trajectory["time"],
                    steps_per_timestep=100
                )
            # TODO: HACK don't hardcode object model name
            learned_system.add_trajectories(
                traj_lens=[len(plant_states_dict.squeeze())],
                traj_data=[plant_states_dict.squeeze()["cube_state"]],
            )
            """
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
            obs_info_inv = None



        elif command_char == "s":
            print("Sampling random action...")
            selected_action = sample_action(workspace_z_rot = 3 * np.pi / 4.0, workspace_radius = 0.15, sphere_radius=0.0175, fixed_240_W=init_trifinger_state[6:9])
            
        elif command_char == "c":
            print("Sampling random action...")
            selected_action = sample_action(workspace_z_rot = 3 * np.pi / 4.0, workspace_radius = 0.15, sphere_radius=0.0175, fixed_240_W=init_trifinger_state[6:9],
                                            library=ActionLibrary.CORNERSINGLE)

        elif command_char == "a":
            print("Recording Inverse Observed Info")
            if obs_info_inv is None:
                obs_info = learned_system.observed_info(traj_dataloader, get_loss_args)
                obs_info_inv = torch.linalg.inv(obs_info)

            print(f"Previously Observed Information: {obs_info}")

            print(f"Sampling {n_actions_optimized} actions to optimize...")
            action_samples = torch.stack([
                torch.vstack([torch.from_numpy(action).clone().to(torch.get_default_device()) for action in sample_action(library=libaction)])
                #for _ in range(n_actions_optimized)
                for libaction in [ActionLibrary.XPINCH]
            ])
            interpolated_actions, timestamps = interpolate_sampled_action(action_samples)
            robot_trajectories = extract_robot_trajectory(learned_system, interpolated_actions, robot_model_name)
            fishers = learned_system.expected_fisher_info(robot_trajectories, timestamps, robot_model_name)
            fishers_obs_weighted = torch.matmul(fishers, obs_info_inv)
            fishers_traces = torch.vmap(torch.trace)(fishers_obs_weighted)
            best_action = action_samples[torch.argmax(fishers_traces)]
            print(f"Best Action Fisher: {fishers[torch.argmax(fishers_traces)]}")
            selected_action = (best_action[0, :].detach().cpu().numpy(), best_action[1, :].detach().cpu().numpy())

        elif command_char == "o":
            if traj_dataloader is None or len(traj_dataloader) == 0:
                print("Data required for observed info\n")
                continue

            obs_info = learned_system.observed_info(traj_dataloader, get_loss_args)

        elif command_char == "t":
            if traj_dataloader is None or len(traj_dataloader) == 0:
                print("Cannot train without data.\n")
                continue
            else:
                print("Training on ", len(traj_dataloader), "trajectories.")

            try:
                epochs = int(input("How many epochs? "))
            except ValueError:
                print("Cancelling...")
                continue
            print("Training...")

            ### TODO: HACK don't repeat vis code
            # TODO: HACK don't hardcode object name
            object_name = "cube"
            true_pose = np.array([1., 0., 0., 0., 0., 0., 0.])

            if new_trajectory is not None and len(new_trajectory) >= 1:
                true_pose = new_trajectory[object_name]["position"][-1].detach().cpu().numpy()

            start_time = time.time()
            for idx in range(epochs):
                train_loss, loss_data = train_epoch(traj_dataloader, learned_system, optimizer)
                total_epochs += 1
                print(total_epochs, 
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
            

            print(f"Finished training {epochs} epochs in {time.time()-start_time} seconds!")
            obs_info_inv = None

        elif command_char == "v":
            print("Visualizing")
            gui_vis.sweep()

            if cube_state is None: #hasn't been declared yet
                last_cube_state_traj = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0325])
            else:
                last_cube_state_traj = cube_state[-1]
            
            
            visualize_geometries(vis_meshcat, learned_system, true_geometry, last_cube_state_traj)
            print("Learned Geometry", learned_system.get_learned_geometry())

            obs_info_inv = None

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
