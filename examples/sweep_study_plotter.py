import glob
import os
from typing import List, Dict, Callable, Tuple

import torch
from matplotlib import pyplot as plt
from torch import Tensor

import wandb
from dair_pll import file_utils
from dair_pll.drake_experiment import MultibodyLosses
from dair_pll.geometry import MeshRepresentation
from param_error_study import MESH_REPRESENTATIONS, LOSSES, \
    name_from_mesh_loss, STUDY_NAME_PREFIX, WANDB_PROJECT
from plot_styler import PlotStyler

#MESH_REPRESENTATIONS = reversed(MESH_REPRESENTATIONS)
storage_name = STUDY_NAME_PREFIX
study_names = [
    name_from_mesh_loss(mesh, loss)
    for mesh in MESH_REPRESENTATIONS
    for loss in LOSSES
]  # type: List[str]

def display_name_from_mesh_loss(mesh_representation: MeshRepresentation,
                                loss: MultibodyLosses) -> str:
    """Generate a name for a sweep instance."""
    mesh_display = {
        MeshRepresentation.POLYGON: "Polygon",
        MeshRepresentation.DEEP_SUPPORT_CONVEX: "DNN"
    }[mesh_representation]
    loss_display = {
        MultibodyLosses.PREDICTION_LOSS: "Prediction Error",
        MultibodyLosses.CONTACTNETS_ANITESCU_LOSS: "Implicit Loss (Ours)",
        MultibodyLosses.CONTACTNETS_NCP_LOSS: "ContactNets"
    }[loss]
    #return f'{mesh_display} + {loss_display}'
    return loss_display

display_names = [
    display_name_from_mesh_loss(mesh, loss)
    for mesh in MESH_REPRESENTATIONS
    for loss in LOSSES
]

scatter_name = 'parameter_relative_error'
#scatter_name = 'valid_trajectory_mse'
scatter_value = 64
WANDB_ENTITY = 'mshalm95'
ANY_INT = '[0-9]*'
PLOT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
file_utils.assure_created(PLOT_DIR)


def get_storage(storage_name: str) -> str:
    return os.path.join(os.path.dirname(__file__), 'storage', storage_name)


def find_sweep_instances(storage_name: str,
                         study_name: str) -> Dict[int, List[str]]:
    storage = get_storage(storage_name)
    sweep_instance_regex = file_utils.sweep_run_name(study_name, '*', '*')
    run_glob_regex = os.path.join(file_utils.all_runs_dir(storage),
                                  sweep_instance_regex)
    all_run_dirs = glob.glob(run_glob_regex)  # type: List[str]
    sweep_instances = {}  # type: Dict[int, List[str]]
    for run_dir in all_run_dirs:
        value = int(run_dir.split("_")[-1])
        if not value in sweep_instances:
            sweep_instances[value] = []
        sweep_instances[value].append(os.path.basename(run_dir))
    return sweep_instances


def recover_wandb_run_id(storage_name: str, run_name: str) -> str:
    storage = get_storage(storage_name)
    checkpoint_file = file_utils.get_training_state_filename(storage, run_name)
    checkpoint = torch.load(checkpoint_file)
    wandb_run_id_base = checkpoint['wandb_run_id']
    if wandb_run_id_base is None:
        raise ValueError('wandb id not stored for run %s' % run_name)
    wandb_run_id = f'{WANDB_ENTITY}/{WANDB_PROJECT}/{wandb_run_id_base}'
    return wandb_run_id


def get_wandb_scatter_entries(
        run_ids: List[str],
        scatter: str,
        direction: Callable = min) -> Tuple[Tensor, Tensor]:
    initial_values = []
    best_values = []
    for run_id in run_ids:
        #pdb.set_trace()
        run = wandb.Api().run(run_id)
        history = run.scan_history(keys=[scatter])
        scatter_values = [float(entry[scatter]) for entry in history]
        initial_values.append(scatter_values[0])
        best_values.append(direction(scatter_values))

    return torch.tensor(initial_values), torch.tensor(best_values)

def wandb_comparison_scatter():
    styler = PlotStyler()
    styler.set_default_styling(directory=PLOT_DIR)
    plt.figure()
    color_map = [styler.green, styler.red]
    for i, study_name in enumerate(study_names):
        sweep_instances = find_sweep_instances(storage_name, study_name)
        scatter_sweep_instances = sweep_instances[scatter_value]
        scatter_wandb_run_ids = [
            recover_wandb_run_id(storage_name, run_name)
            for run_name in scatter_sweep_instances
        ]

        init_scatter, min_scatter = get_wandb_scatter_entries(
            scatter_wandb_run_ids, scatter_name)
        #print(display_name, init_scatter, min_scatter)

        # draw matplotlib scatter plot with init_scatter as the x-axis and
        # min_scatter as the y-axis.
        plt.scatter(init_scatter, min_scatter, c=color_map[i],
                    zorder=1)

    # set log axes
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel('Initial Parameter Relative Error')
    plt.ylabel('Final Parameter Relative Error')
    plt.title('Improvement in Parameter Estimation over Training')
    plt.legend(display_names)
    #pdb.set_trace()
    xmin, xmax, ymin, ymax = plt.axis()
    # plot y = x line
    min_min = min(xmin, ymin)
    max_max = max(xmax, ymax)
    plt.plot([min_min, max_max], [min_min, max_max], c='black',zorder=0)
    plt.xlim(min_min, max_max)
    plt.ylim(min_min, max_max)

    #plt.show()
    styler.save_fig("wandb_comparison_scatter.png")

if __name__ == '__main__':
    wandb_comparison_scatter()




