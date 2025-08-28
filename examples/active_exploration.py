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

from copy import deepcopy
import os
import pdb
import signal
import sys
import time
from typing import Type, Optional
from functools import partial

import gin
import gin.torch.external_configurables
import git
import numpy as np
from tensordict import TensorDict
import torch

from dair_pll import file_utils, action_utils, experiment_utils
from dair_pll.dataset_management import TrajectorySet
from dair_pll.gui_utils import PLLMeshcatVisualizer
from dair_pll.hack_utils import extract_robot_trajectory
from dair_pll.multibody_tactile_learnable_system import MultibodyLearnableTactileSystem
from dair_pll.trifinger_utils import TrifingerLCMService

# Repository directory (default for file operations)
REPO_DIR = os.path.normpath(
    git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")
)
DEFAULT_CONFIG = "active_exploration.gin"


## Main Function
signal_pressed = False


def signal_handler(_sig, _frame):
    """Handle SIGINT"""
    # pylint: disable=global-statement
    global signal_pressed
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal_pressed = True
    signal.signal(signal.SIGINT, signal_handler)


@gin.configurable
def main(
    storage_folder_name: str = "storage_active",
    run_name: str = "default_run",
    optimizer_cls: Type = torch.optim.SGD,
    action_params: action_utils.ActionWorkspaceParams = action_utils.ActionWorkspaceParams(),
):
    """Main function for online learning loop"""
    ### Signal Handling
    # pylint: disable=global-statement
    global signal_pressed
    signal_pressed = False

    signal.signal(signal.SIGINT, signal_handler)
    # torch.autograd.set_detect_anomaly(True) ## NOTE: doesn't work with vmap
    # Debug: Remove scientific notation for numpy printing
    np.set_printoptions(suppress=True)
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float64)

    # Create run directory
    print("Active Tactile Exploration")
    storage_name = os.path.join(REPO_DIR, "results", storage_folder_name)
    print(f"Storing data and results at {file_utils.run_dir(storage_name, run_name)}")

    # Initialize LCM
    trifinger_lcm = TrifingerLCMService()
    print("Resetting Trifinger Position...")
    trifinger_lcm.execute_trajectory(action_params.get_reset_knot(), no_data=True)
    new_trajectory = None

    # Create learnable system
    print("Loading Learned System...")
    # Pylint doesn't know about gin
    # pylint: disable=no-value-for-parameter
    learned_system = MultibodyLearnableTactileSystem()

    # Create Dataset
    data_trajectories = TrajectorySet()

    # GUI Visualization
    true_geom, true_mesh = experiment_utils.get_true_geometry_and_mesh()
    gui_vis = PLLMeshcatVisualizer(
        system=learned_system,
        data=data_trajectories,
        true_geom=true_geom,
        true_pose=trifinger_lcm.get_current_object_pose(),
    )

    # Sample initial action (from true obj pose)
    action_cem = action_utils.ActionCEM()
    selected_action = action_cem.random_action()
    selected_knots = np.stack(
        [action_params.get_reset_knot(), action_params.get_reset_knot()]
    )
    first_knots = action_utils.action_to_knots(
        action_params,
        [selected_action],
        trifinger_lcm.get_current_object_pose(),
        force_finger=0,
    )[0]
    # Only move one finger
    selected_knots[:, :3] = first_knots[:, :3]
    gui_vis.draw_action_samples(selected_knots[np.newaxis, :, :])

    # Initialize Optimizer and Data config
    optimizer = optimizer_cls(learned_system.parameters())
    total_epochs = 0
    cham_dists = []

    ## Visualizing Action Samples
    def action_vis(actions):
        print("Visualizing...")
        obj_pose_guess = learned_system.get_learned_centroid()
        knots = action_utils.action_to_knots(action_params, actions, obj_pose_guess)
        gui_vis.draw_action_samples(knots)

    ### Function Definitions
    def reset_robot(non_blocking=True):
        nonlocal action_params, trifinger_lcm
        print("Resetting Trifinger Position...")
        trifinger_lcm.execute_trajectory(
            action_params.get_reset_knot(), no_data=True, non_blocking=non_blocking
        )

    force_finger = 1

    def select_action():
        nonlocal force_finger, selected_action, selected_knots, gui_vis, action_params, trifinger_lcm, learned_system, data_trajectories
        # score_fn = experiment_utils.score_random
        score_fn = partial(
            experiment_utils.score_eig,
            action_params,
            learned_system,
            data_trajectories,
            trifinger_lcm,
            force_finger,
        )
        selected_action = action_cem.best_action(score_fn=score_fn, vis_fn=action_vis)
        obj_pose_guess = learned_system.get_learned_centroid()
        selected_knots = action_utils.action_to_knots(
            action_params,
            [selected_action],
            obj_pose_guess,
            force_finger=force_finger,
        )[0]
        gui_vis.draw_action_samples(selected_knots[np.newaxis, :, :])
        force_finger = 1 - force_finger

    def collect_data(sim_overwrite: bool = False):
        nonlocal total_epochs, learned_system, optimizer, data_trajectories, gui_vis, selected_action, selected_knots, trifinger_lcm
        ## Execute selected action
        # Move to start state
        trifinger_lcm.execute_trajectory(selected_knots[0], no_data=True)
        time.sleep(0.1)

        # Execute and collect data
        new_trajectory = trifinger_lcm.execute_trajectory(selected_knots[1])

        # Move back to start state
        trifinger_lcm.execute_trajectory(
            selected_knots[0], no_data=True, non_blocking=True
        )

        if len(new_trajectory) < 1:
            print("WARNING: No data collected")
            return

        # Add data to dataset
        first_contact = len(
            new_trajectory["time"]
        )  # if len(data_trajectories.trajectories) == 0 else 0
        for finger_name in new_trajectory.keys():
            try:
                test_firstcontact = int(
                    torch.nonzero(
                        torch.linalg.vector_norm(
                            new_trajectory[finger_name]["contact_normal_W"], dim=-1
                        )
                    )[0]
                )
                if test_firstcontact < first_contact:
                    first_contact = test_firstcontact
            except (IndexError, KeyError):  # e.g. object, time
                continue
        # Minimize ground truth teleportation
        if first_contact > 0:
            first_contact -= 1
        # Get entire desired trajectory
        desired_traj = extract_robot_trajectory(
            action_utils.interpolate_sampled_action(
                data=torch.tensor(np.array(selected_knots)),
                traj_len_s=(new_trajectory["time"][-1] - new_trajectory["time"][0]),
                traj_n_steps=len(new_trajectory),
            )[0],
            learned_system,
            trifinger_lcm,
        )
        # Crop to first contact
        new_trajectory = new_trajectory[first_contact:]
        add_trajectory = TensorDict({}, batch_size=new_trajectory.batch_size)
        add_trajectory["time"] = new_trajectory["time"]
        add_trajectory[learned_system.controlled_model_names[0] + "_state"] = (
            extract_robot_trajectory(new_trajectory, learned_system, trifinger_lcm)
        )
        add_trajectory[learned_system.controlled_model_names[0] + "_desired"] = (
            desired_traj[first_contact:]
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

        if sim_overwrite:
            print("Running Sim...")
            learned_system.learned_trajectory_sim_overwrite(data_trajectories)

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

    def train_on_data(
        n_epochs: Optional[int] = None, patience: int = 200, n_sim_overwrite: int = 300
    ):
        nonlocal total_epochs, learned_system, optimizer, data_trajectories, gui_vis, cham_dists
        global signal_pressed
        if len(data_trajectories.trajectories) == 0:
            print("Cannot train without data.\n")
            return

        if n_epochs == None:
            try:
                epochs = int(input("How many epochs? "))
            except ValueError:
                print("Cancelling...")
                return
            print("Training...")
        else:
            epochs = n_epochs

        start_time = time.time()
        meas_contact_forces = {
            (learned_system.get_learned_body_name(), str(k)): v
            for k, v in data_trajectories.get_full_trajectory(
                key="contact_forces"
            ).items()
        }
        meas_contact_normals = {
            (learned_system.get_learned_body_name(), str(k)): v
            for k, v in data_trajectories.get_full_trajectory(
                key="contact_normals"
            ).items()
        }
        timestamps = data_trajectories.get_full_trajectory(key="time")
        best_loss = 0.0
        epochs_since_best = 0
        best_state = None
        did_sim = False
        do_sim = False
        for idx in range(epochs):
            optimizer.zero_grad()

            # Overwrite with simulation
            if do_sim:
                do_sim = False
                did_sim = True
                learned_system.learned_trajectory_sim_overwrite(data_trajectories)
                # Reset Patience
                epochs_since_best = 0

            forward_args = learned_system(
                ctrl_desired=data_trajectories.get_full_trajectory(
                    key=learned_system.controlled_model_names[0] + "_desired"
                ),
                timestamps=data_trajectories.get_full_trajectory(key="time"),
                ctrl_actual=data_trajectories.get_full_trajectory(
                    key=learned_system.controlled_model_names[0] + "_state"
                ),
            )

            gui_vis.learned_plant_traj = forward_args[0]
            gui_vis.update()

            loss_dict = learned_system.loss_fn(
                None, meas_contact_normals, timestamps, *forward_args
            )
            loss_total = sum(torch.sum(v) for _, v in loss_dict.items())
            loss_total.backward()
            optimizer.step()

            loss_print = tuple(
                f"\t{k}: {torch.sum(v).detach().cpu()};\n" for k, v in loss_dict.items()
            )

            total_epochs = total_epochs + 1
            print(total_epochs, f"Loss: {loss_total:.3e};\n", *loss_print)
            if idx == 0 or float(loss_total.detach().cpu().numpy()) < best_loss:
                best_loss = float(loss_total.detach().cpu().numpy())
                epochs_since_best = 0
                best_state = deepcopy(learned_system.state_dict())
            else:
                epochs_since_best += 1
                if epochs_since_best >= patience:
                    if did_sim:
                        print("Patience ran out, exiting...")
                        break
                    else:
                        print("Patience ran out, trying sim...")
                        do_sim = True
                        continue

            print(f"Quit Training Signal?: {signal_pressed}")
            if signal_pressed:
                signal_pressed = False
                print("Training cancelled...")
                breakpoint()
                epochs = idx + 1
                break

        if best_state is not None:
            learned_system.load_state_dict(best_state)

            ## Reinit Visualization
            forward_args = learned_system(
                ctrl_desired=data_trajectories.get_full_trajectory(
                    key=learned_system.controlled_model_names[0] + "_desired"
                ),
                timestamps=data_trajectories.get_full_trajectory(key="time"),
                ctrl_actual=data_trajectories.get_full_trajectory(
                    key=learned_system.controlled_model_names[0] + "_state"
                ),
            )

            gui_vis.learned_plant_traj = forward_args[0]
            gui_vis.update()

        print(f"Finished training {epochs} epochs in {time.time()-start_time} seconds!")

    def report_chamfer_dist():
        cham_dist = experiment_utils.chamfer_metric(
            learned_system, true_mesh, trifinger_lcm.get_current_object_pose()
        )
        print(f"Chamfer Distance (m): {cham_dist}")
        cham_dists.append(cham_dist)

    ## Start Input Loop
    def print_help():
        print(
            "\nUsage:\n"
            "a - Action Selecton\n"
            "e - Execute selected action + collect data\n"
            "s - Sample random action\n"
            "t - Train\n"
            "b - breakpoint()\n"
            "o - DEBUGGING COMMAND\n"
            "v - Visualize\n"
            "r - Reset Trifinger\n"
            "h - Print Help\n"
            "q - Quit\n"
        )

    print_help()
    command_char = " "
    while command_char != "q":
        command_char = input("Command $ ").split(" ")[0]

        if command_char == "h":
            print_help()

        elif command_char == "f":
            print("Running full experiment")

            reset_robot()

            max_iter = 10
            for idx in range(max_iter):
                print("Collecting Data...")
                reset_robot(non_blocking=False)
                collect_data()
                print("Visualizing...")
                gui_vis.sweep()
                print("Training...")
                train_on_data(n_epochs=500)
                print("Record Chamfer Distance...")
                report_chamfer_dist()
                print(f"Chamfer Distances So Far: {cham_dists}")
                if idx == (max_iter - 1):
                    break
                print("Select next action...")
                select_action()

        elif command_char == "b":
            # pylint: disable-next=forgotten-debug-statement
            pdb.Pdb(nosigint=True).set_trace()

        elif command_char == "r":
            reset_robot()

        elif command_char == "a":
            select_action()

        elif command_char == "e":
            collect_data()

        elif command_char == "s":
            try:
                action_params = [
                    float(x)
                    for x in input(
                        "Enter Action Params (0RA 0Dec 120RA 120Dec): "
                    ).split()
                ]
            except ValueError:
                print("Cancelling...")
                continue

            selected_action = action_utils.Action(*action_params)
            selected_knots = action_utils.action_to_knots(
                action_params, [selected_action], true_obj_pose
            )[0]
            gui_vis.draw_action_samples(selected_knots[np.newaxis, :, :])

        elif command_char == "v":
            print("Visualizing entire trajectory.")
            gui_vis.sweep()
            print("Done!")

        elif command_char == "o":
            """DEBUGGING COMMAND"""
            if len(data_trajectories.trajectories) == 0:
                print("Need data for sim overwrite.\n")
                return
            learned_system.learned_trajectory_sim_overwrite(data_trajectories)
            forward_args = learned_system(
                ctrl_desired=data_trajectories.get_full_trajectory(
                    key=learned_system.controlled_model_names[0] + "_desired"
                ),
                timestamps=data_trajectories.get_full_trajectory(key="time"),
                ctrl_actual=data_trajectories.get_full_trajectory(
                    key=learned_system.controlled_model_names[0] + "_state"
                ),
            )

            gui_vis.learned_plant_traj = forward_args[0]
            gui_vis.update()

        elif command_char == "t":
            train_on_data()

    # Quit


def main_fn():
    """Entry point"""
    config_file = DEFAULT_CONFIG
    if len(sys.argv) < 2:
        print(f"Warning: Using default config file ({DEFAULT_CONFIG})")
    else:
        config_file = sys.argv[1]

    # Parse config file and start
    gin.register(np.array, module="np")
    gin.parse_config_file(os.path.join(REPO_DIR, "config", config_file))
    # Pylint doesn't know about gin
    # pylint: disable=no-value-for-parameter
    main()


if __name__ == "__main__":
    main_fn()
