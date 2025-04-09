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

# tensor_utils required for TensorDict's collate_fn
# pylint: disable-next=unused-import
from dair_pll import tensor_utils
from dair_pll import file_utils
from dair_pll.drake_system import DrakeSystem
from dair_pll.dataset_management import TrajectorySet
from dair_pll.gui_utils import PLLMeshcatVisualizer
from dair_pll.multibody_learnable_system import MultibodyLearnableSystemWithTrajectory

from dair_pll.tensor_utils import pbmm
from dair_pll.hack_utils import finger_idx_from_body_name

# relative imports
from trifinger_lcm_service import TrifingerLCMService
from action_library import ActionLibrary, sample_action

# Repository directory (default for file operations)
REPO_DIR = os.path.normpath(
    git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel")
)
DEFAULT_CONFIG = "rss_experiment.gin"

torch.set_default_device("cuda")

# global signal_pressed
# signal.signal(signal.SIGINT, signal_handler)
#torch.autograd.set_detect_anomaly(True) ## NOTE: doesn't work with vmap
torch.set_default_device("cuda")


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

@gin.configurable(denylist=["data"])
def interpolate_sampled_action(
    data: Tensor, fingertip_body_names: List[str], traj_len_s=2.0, traj_n_steps=61
) -> TensorDictBase:
    """
    Interpolates a start/end action using a cubic spline.

    Params:
        data: outputs of sample_action size (batch, 2, 18)

    Returns:
        TensorDict input of extract_robot_trajectory, batch_size=(batch, traj_n_steps), keys = fingertip_body_names
        Timestamps = Tensor size (traj_n_steps,)
    """
    assert len(data.size()) >= 2
    batch_dims = data.size()[:-2]
    assert data.size() == batch_dims + (2, 18)
    ret = TensorDict({}, batch_size=batch_dims + (traj_n_steps,))
    ret_timestamps = torch.linspace(0.0, traj_len_s, traj_n_steps) # (traj_n_steps,)
    rel_timestamps = ((ret_timestamps - ret_timestamps[0]) / (ret_timestamps[-1] - ret_timestamps[0])).unsqueeze(0) # (1, traj_n_steps)
    samples = data[..., :, :9] # (batch, 2, 9)
    samples_dot = data[..., :, 9:] # (batch, 2, 9)
    spline_a = samples[..., 0, :].unsqueeze(-1) # (batch, 9, 1)
    spline_b = samples_dot[..., 0, :].unsqueeze(-1) # (batch, 9, 1)
    spline_c = (3.*(samples[..., 1, :]-samples[..., 0, :]) - 2.*samples_dot[..., 0, :] - samples_dot[..., 1, :]).unsqueeze(-1) # (batch, 9, 1)
    spline_d = (2.*(samples[..., 0, :]-samples[..., 1, :]) + samples_dot[..., 0, :] + samples_dot[..., 1, :]).unsqueeze(-1) # (batch, 9, 1)

    # (batch, traj_n_steps, 9)
    data_lerp = spline_a @ torch.pow(rel_timestamps, 0.) + spline_b @ torch.pow(rel_timestamps, 1.) + spline_c @ torch.pow(rel_timestamps, 2.) + spline_d @ torch.pow(rel_timestamps, 3.)
    data_lerp = torch.transpose(data_lerp, -1, -2)
    data_lerp_dot = spline_b @ torch.pow(rel_timestamps, 0.) + 2.*spline_c @ torch.pow(rel_timestamps, 1.) + 3.*spline_d @ torch.pow(rel_timestamps, 2.)
    data_lerp_dot = torch.transpose(data_lerp_dot, -1, -2)

    for fingertip in fingertip_body_names:
        ret[fingertip, "position"] = torch.zeros(batch_dims + (traj_n_steps, 3))
        ret[fingertip, "velocity"] = torch.zeros(batch_dims + (traj_n_steps, 3))
    for finger_idx, fingertip in enumerate(fingertip_body_names):
        pos_idx = 3 * finger_idx
        ret[fingertip, "position"][..., :] = data_lerp[..., pos_idx : pos_idx + 3]
        ret[fingertip, "velocity"][..., :] = data_lerp_dot[..., pos_idx : pos_idx + 3]

    return ret, ret_timestamps

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

    for finger_name, state_idx in zip(
        fingertip_body_names,
        finger_idx_from_body_name(
            plant, plant.GetModelInstanceByName(robot_model_name), fingertip_body_names
        ),
    ):
        pos_idx = 3 * state_idx
        vel_idx = robot_space.n_x // 2 + pos_idx
        ret[..., pos_idx : pos_idx + 3] = data[finger_name]["position"]
        ret[..., vel_idx : vel_idx + 3] = data[finger_name]["velocity"]

    return ret

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
    """ Handle SIGINT"""
    global signal_pressed
    signal_pressed = True



class sim_experiment():

    @gin.configurable
    def __init__(self,
        init_trifinger_state: List[float],
        safe_trifinger_height: float,
        robot_model_name: str,
        object_model_name: str = "cube",
        n_actions_optimized: int = 30,
        storage_folder_name: str = "storage_rss",
        run_name: str = "default_run",
        optimizer_cls: Type = torch.optim.SGD,):

        """Main function for online learning loop"""

        # instance variables
        self.trifinger_lcm_ = trifinger_lcm
        self.safe_trifinger_height_ = safe_trifinger_height
        self.learned_system_ = learned_system
        self.robot_model_name_ = robot_model_name
        self.object_model_name_ = object_model_name
        self.data_trajectories_ = data_trajectories
        self.optimizer_cls_ = optimizer_cls

        self.train_losses_ = train_losses
        self.train_loss_data_ = train_loss_data

        # Create run directory
        print("Hyperparameter Tuning for Active Tactile Exploration")
        storage_name = os.path.join(REPO_DIR, "results", storage_folder_name)
        print(f"Storing data and results at {file_utils.run_dir(storage_name, run_name)}")

        # Create learnable system
        print("Loading Learned System...")
        learned_system = MultibodyLearnableSystemWithTrajectory(
            output_urdfs_dir=file_utils.get_learned_urdf_dir(storage_name, run_name)
        )
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
        trifinger_lcm = TrifingerLCMService()
        print("Move to initial trifinger state")
        trifinger_lcm.execute_trajectory(np.array(init_trifinger_state), no_data=True)
        print("Sample Initial Random Action...")
        self.selected_action = sample_action(library=ActionLibrary.XSINGLE)
        new_trajectory = None

        # Initialize Optimizer and Data config
        optimizer = optimizer_cls(learned_system.parameters())
        traj_dataloader = None
        obs_info_inv = None
        total_epochs = 0


    def set_weights(self,
                    weights: Dict[str, float],
                    ) -> None:
        """Set weights for loss function"""


    def data_collection(self,
              selected_action,):
        """Method for running a single trial
            - for a given intial pose of the object
            - execute each action in the action library
            - collect data and train for a given number of epochs
        """

        trifinger_lcm = self.trifinger_lcm_
        safe_trifinger_height = self.safe_trifinger_height_
        learned_system = self.learned_system_
        robot_model_name = self.robot_model_name_
        object_model_name = self.object_model_name_
        data_trajectories = self.data_trajectories_
        optimizer_cls = self.optimizer_cls_

        # Move to start state
        trifinger_lcm.execute_trajectory(selected_action[0], no_data=True)

        # Execute and collect data
        new_trajectory = trifinger_lcm.execute_trajectory(selected_action[1])

        if len(new_trajectory) < 1:
            print("WARNING: No data collected")

        # Move straight up
        safe_state = np.copy(selected_action[0])
        safe_state[:3] = (new_trajectory["finger_0"]["position"][-1].cpu().clone().numpy())
        safe_state[2] = safe_trifinger_height
        safe_state[3:6] = (new_trajectory["finger_1"]["position"][-1].cpu().clone().numpy())
        safe_state[5] = safe_trifinger_height
        self.trifinger_lcm_.execute_trajectory(safe_state, no_data=True)

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

        data_trajectories.add_trajectories([add_trajectory.clone().detach()],
            torch.tensor([len(data_trajectories.trajectories)], dtype=torch.int),
            )

        learned_system.add_trajectories(traj_lens=[len(add_trajectory["time"])])

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

        self.traj_dataloader_ = traj_dataloader

    def data_train(self,
                   epochs: int,
                   ):
        
        traj_dataloader = self.traj_dataloader_
        learned_system = self.learned_system_
        new_trajectory = self.new_trajectory_
        optimizer = self.optimizer_cls_

        train_losses = self.train_losses_
        train_loss_data = self.train_loss_data_
        learned_summaries = self.learned_summaries_

        assert traj_dataloader is not None or len(traj_dataloader) > 0, "Cannot train without data"

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
