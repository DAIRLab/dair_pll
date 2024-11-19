#!/usr/bin/env python3
"""
Run a simulated online learning experiment
"""

import os
import pdb
import signal
import sys
import time
from typing import cast, Any, Dict, List, Type, Optional

import gin
import gin.torch.external_configurables
import git
import numpy as np
import scipy
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tensordict.tensordict import TensorDict

# tensor_utils required for TensorDict's collate_fn
# pylint: disable-next=unused-import
from dair_pll import tensor_utils
from dair_pll import file_utils, drake_controllers
from dair_pll.dataset_management import TrajectorySet
from dair_pll.drake_system import DrakeSystem, carry_dict_create
from dair_pll.multibody_learnable_system import MultibodyLearnableSystemWithTrajectory
from dair_pll import vis_utils


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

    ret = {
        "x": x_past,
        "u": control,
        "x_plus": x_plus,
        "contact_forces": contact_forces,
    }
    if impulses is not None:
        ret["impulses"] = impulses

    return ret

def direct_loss_impulses(traj: np.ndarray, data: DataLoader, system: MultibodyLearnableSystemWithTrajectory):
    """
    Return the impulses
    """
    assert len(system._trajectory_model_names) == 1
    traj_model_name = system._trajectory_model_names[0] + "_state"
    traj_data = [d for d in data][0]
    assert len(traj) == (traj_data[0].shape[0] + 1) * system._trajectory.space.n_x
    traj_state = torch.tensor(np.copy(traj)).reshape((traj_data[0].shape[0] + 1), 1, system._trajectory.space.n_x)
    traj_data[0][traj_model_name] = traj_state[:-1].clone()
    traj_data[1][traj_model_name] = traj_state[1:].clone()

    return system.calculate_contactnets_impulses(**get_loss_args(traj_data[0], traj_data[1], system))

def direct_loss_jacobian(traj: np.ndarray, data: DataLoader, system: MultibodyLearnableSystemWithTrajectory, impulses: Optional[Tensor] = None):
    """
    Direct compute of the loss for the purpose of scipy minimize
    """
    traj_model_name = system._trajectory_model_names[0] + "_state"
    traj_data = [d for d in data][0]
    traj_state = torch.tensor(traj).reshape((traj_data[0].shape[0] + 1), 1, system._trajectory.space.n_x)

    def loss_from_tensor_jac(traj_tensor):
        traj_data[0][traj_model_name] = traj_tensor[:-1]
        traj_data[1][traj_model_name] = traj_tensor[1:]
        return system.contactnets_loss(**get_loss_args(traj_data[0], traj_data[1], system, impulses)).mean()

    return torch.autograd.functional.jacobian(loss_from_tensor_jac, traj_state).flatten().cpu().numpy()

def direct_loss_hessian(traj: np.ndarray, data: DataLoader, system: MultibodyLearnableSystemWithTrajectory, impulses: Optional[Tensor] = None):
    """
    Direct compute of the loss for the purpose of scipy minimize
    """
    traj_model_name = system._trajectory_model_names[0] + "_state"
    traj_data = [d for d in data][0]
    traj_state = torch.tensor(traj).reshape((traj_data[0].shape[0] + 1), 1, system._trajectory.space.n_x)

    def loss_from_tensor_hes(traj_tensor):
        traj_data[0][traj_model_name] = traj_tensor[:-1]
        traj_data[1][traj_model_name] = traj_tensor[1:]
        return system.contactnets_loss(**get_loss_args(traj_data[0], traj_data[1], system, impulses)).mean()

    return torch.autograd.functional.hessian(loss_from_tensor_hes, traj_state).cpu().numpy()

def direct_loss(traj: np.ndarray, data: DataLoader, system: MultibodyLearnableSystemWithTrajectory, impulses: Optional[Tensor] = None):
    """
    Direct compute of the loss for the purpose of scipy minimize
    """
    assert len(system._trajectory_model_names) == 1
    traj_model_name = system._trajectory_model_names[0] + "_state"
    traj_data = [d for d in data][0]
    assert len(traj) == (traj_data[0].shape[0] + 1) * system._trajectory.space.n_x
    traj_state = torch.tensor(traj).reshape((traj_data[0].shape[0] + 1), 1, system._trajectory.space.n_x)
    traj_data[0][traj_model_name] = traj_state[:-1]
    traj_data[1][traj_model_name] = traj_state[1:]

    return float(system.contactnets_loss(**get_loss_args(traj_data[0], traj_data[1], system, impulses)).mean())



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


signal_pressed = False
def signal_handler(sig, frame):
    """ Handle SIGINT"""
    global signal_pressed
    signal_pressed = True

## Main Function
@gin.configurable
def main(
    storage_folder_name: str = "storage",
    run_name: str = "default_run",
    optimizer_cls: Type = torch.optim.SGD,
    use_true_traj: bool = False,
    torch_default_device: str = "cpu",
):
    """Main function for online learning loop"""
    global signal_pressed
    signal.signal(signal.SIGINT, signal_handler)
    # pylint: disable=R0915, R0912, R0914
    # Expect main function to have a lot of statements and branches and variables
    # pylint: disable=E1120
    # Expect gin to handle missing arguments
    torch.set_default_device(torch_default_device)

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

    # Set of Simulated Trajectories
    sim_trajectories = TrajectorySet()

    # Load False URDFs into Learned System
    print("Loading Learned System...")
    learned_system = MultibodyLearnableSystemWithTrajectory(
        output_urdfs_dir=file_utils.get_learned_urdf_dir(storage_name, run_name)
    )
    learned_summaries = [learned_system.summary({})]
    train_losses = []
    train_loss_data = []

    # Set Up Visualization System
    vis_system = None

    # Initialize Optimizer and Data config
    optimizer = optimizer_cls(learned_system.parameters())
    traj_dataloader = None
    total_epochs = 0

    # Start Input Loop
    def print_help():
        print(
            "\nUsage:\n"
            "b - breakpoint()\n"
            "c - Collect Sim Data\n"
            "d - Debug Toggle\n"
            "h - Print Help\n"
            "m - Meshcat Visualize\n"
            "o - Optimize traj directly\n"
            "t - Train\n"
            "u - Update PID Ref\n"
            "v - Visualize\n"
            "q - Quit\n"
        )

    print_help()
    command_char = " "
    while command_char != "q":
        command_char = input("Command $ ").split(" ")[0]

        if command_char == "h":
            print_help()

        elif command_char == "o":
            if traj_dataloader is None:
                print("Cannot optimize without sim data.\n")
                continue

            init_state = np.array([0., 0.05, 0., 0., 0., 0.])
            traj_0 = np.tile(init_state, len(sim_trajectories.slices) + 1)

            #impulses = direct_loss_impulses(traj_0, traj_dataloader, learned_system)

            res = scipy.optimize.minimize(direct_loss, traj_0, tol=1e-10, args=(traj_dataloader, learned_system), jac=direct_loss_jacobian, method='L-BFGS-B', options={"iprint": 100})
            breakpoint()

        elif command_char == "b":
            # pylint: disable-next=forgotten-debug-statement
            pdb.Pdb(nosigint=True).set_trace()

        elif command_char == "d":
            learned_system.set_debug(not learned_system.debug)
            if learned_system.debug:
                print("Debug enabled")
            else:
                print("Debug disabled")

        elif command_char == "c":
            seconds = float(input("How long (s)? "))
            if seconds <= 0.0:
                continue
            state, data = base_system.simulate(
                initial_state, carry_dict, int(seconds / base_system.dt)
            )
            data["state"] = state.unsqueeze(-2)
            if len(sim_trajectories.trajectories) > 0:
                # Adjust time to be after previous trajectory
                if "time" in data:
                    data["time"] += sim_trajectories.trajectories[-1]["time"][-1]
            sim_trajectories.add_trajectories(
                [data.clone().detach()],
                torch.tensor([len(sim_trajectories.trajectories)], dtype=torch.int),
            )

            # Extend Learnable Trajectory
            learned_system.add_trajectories(
                traj_lens=[state.shape[0]],
                traj_data=[learned_system.model_states_from_state_tensor(state)[learned_system._trajectory_model_names[0] + "_state"]] if use_true_traj else None,
            )

            # Re-init optimizer and data-loader
            batch_size = (
                len(sim_trajectories.slices)
                if sim_trajectories.slices.config.batch_size == -1
                else sim_trajectories.slices.config.batch_size
            )
            traj_dataloader = DataLoader(
                sim_trajectories.slices,
                batch_size=batch_size,
                shuffle=sim_trajectories.slices.config.shuffle,
                generator=torch.Generator(device=torch.get_default_device()),
            )
            optimizer = optimizer_cls(learned_system.parameters())

            # Update Initial State
            initial_state = state[-1:, :]

        elif command_char == "t":
            if traj_dataloader is None:
                print("Cannot train without sim data.\n")
                continue

            try:
                epochs = int(input("How many epochs? "))
            except ValueError:
                print("Cancelling...")
                continue
            print("Training...")

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
                )
                train_losses.append(train_loss)
                train_loss_data.append(loss_data)
                learned_summaries.append(learned_system.summary({}))
                if signal_pressed:
                    signal_pressed = False
                    print("Training cancelled...")
                    epochs = idx + 1
                    break

            vis_system = None  # Invalidate

            print(f"Finished training {epochs} epochs in {time.time()-start_time} seconds!")

        elif command_char == "u":
            print("Enter comma-space-separated floats.\n")
            try:
                updated_ref = np.array(list(map(float, input("New State: ").split(", "))))
            except ValueError:
                print("Could not interpret as comma-space-separated float list")
                continue
            try:
                drake_controllers.update_pid_reference(base_system, updated_ref)
            except AssertionError:
                print("Incorrect state size.")

        elif command_char in ("v", "m"):
            if traj_dataloader is None:
                print("Cannot visualize without sim data.\n")
                continue

            # (Re)Create Vis System
            recreate_vis = (vis_system is None) or (bool(command_char == "m") != vis_system.plant_diagram.vis_is_meshcat())
            if recreate_vis:
                vis_system = vis_utils.generate_visualization_system(
                    base_system=base_system,
                    learned_system=DrakeSystem(
                        urdfs=learned_system.generate_updated_urdfs("vis"),
                        dt=base_system.dt,
                        visualization_file=None,
                    ),
                    visualization_file=(
                        "meshcat"
                        if command_char == "m"
                        else file_utils.get_trajectory_video_filename(
                            storage_name, run_name, total_epochs
                        )
                    ),
                )

            # Visualize joint Trajectory
            joint_traj = torch.cat(
                [
                    base_system.model_states_from_state_tensor(traj["state"])
                    for traj in sim_trajectories.trajectories
                ]
            )
            learned_traj = torch.cat(
                [
                    learned_system.model_states_from_state_tensor(
                        learned_system.construct_state_tensor(traj),
                        vis_utils.LEARNED_TAG,
                    )
                    for traj in sim_trajectories.trajectories
                ]
            )

            for key, val in learned_traj.items():
                joint_traj[key] = val

            joint_traj = vis_system.construct_state_tensor(joint_traj).squeeze(-2)
            vis_utils.visualize_trajectory(vis_system, joint_traj)
            if command_char == "v":
                print(
                    f"Trajectory Written to: {file_utils.get_trajectory_video_filename(storage_name, run_name, total_epochs)}"
                )

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
