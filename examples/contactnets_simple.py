"""Simple ContactNets/differentiable physics learning examples."""
# pylint: disable=E1103
import os
import time
from typing import cast

import sys
import pdb

import click
import numpy as np
import torch
from torch import Tensor
import pickle
import git

from dair_pll import file_utils
from dair_pll.dataset_generation import DataGenerationConfig, \
    ExperimentDatasetGenerator
from dair_pll.dataset_management import DataConfig, \
    TrajectorySliceConfig
from dair_pll.drake_experiment import \
    DrakeMultibodyLearnableExperiment, DrakeSystemConfig, \
    MultibodyLearnableSystemConfig, MultibodyLosses, \
    DrakeMultibodyLearnableExperimentConfig
from dair_pll.experiment import default_epoch_callback
from dair_pll.experiment_config import OptimizerConfig
from dair_pll.hyperparameter import Float, Int
from dair_pll.multibody_learnable_system import MultibodyLearnableSystem, \
    LOSS_PLL_ORIGINAL, LOSS_INERTIA_AGNOSTIC, LOSS_BALANCED, LOSS_POWER, \
    LOSS_VARIATIONS, LOSS_VARIATION_NUMBERS
from dair_pll.state_space import UniformSampler, GaussianWhiteNoiser
from dair_pll.system import System


# Possible systems on which to run PLL
CUBE_SYSTEM = 'cube'
ELBOW_SYSTEM = 'elbow'
SYSTEMS = [CUBE_SYSTEM, ELBOW_SYSTEM]

# Possible dataset types
SIM_SOURCE = 'simulation'
REAL_SOURCE = 'real'
DYNAMIC_SOURCE = 'dynamic'
DATA_SOURCES = [SIM_SOURCE, REAL_SOURCE, DYNAMIC_SOURCE]

# Possible inertial parameterizations to learn for the elbow system.
# The options are:
# 0 - none (0 parameters)
# 1 - masses (n_bodies - 1 parameters)
# 2 - CoMs (3*n_bodies parameters)
# 3 - CoMs and masses (4*n_bodies - 1 parameters)
# 4 - all (10*n_bodies - 1 parameters)
INERTIA_PARAM_CHOICES = [str(i) for i in range(5)]
INERTIA_PARAM_DESCRIPTIONS = [
    'learn no inertial parameters (0 * n_bodies)',
    'learn only masses and not the first mass (n_bodies - 1)',
    'learn only centers of mass (3 * n_bodies)',
    'learn masses (except first) and centers of mass (4 * n_bodies - 1)',
    'learn all parameters (except first mass) (10 * n_bodies - 1)']
INERTIA_PARAM_OPTIONS = ['none', 'masses', 'CoMs', 'CoMs and masses', 'all']


# File management.
CUBE_DATA_ASSET = 'contactnets_cube'
ELBOW_DATA_ASSET = 'contactnets_elbow'
CUBE_BOX_URDF_ASSET = 'contactnets_cube.urdf'
CUBE_MESH_URDF_ASSET = 'contactnets_cube_mesh.urdf'
ELBOW_BOX_URDF_ASSET = 'contactnets_elbow.urdf'
ELBOW_MESH_URDF_ASSET = 'contactnets_elbow_mesh.urdf'

TRUE_DATA_ASSETS = {CUBE_SYSTEM: CUBE_DATA_ASSET, ELBOW_SYSTEM: ELBOW_DATA_ASSET}

MESH_TYPE = 'mesh'
BOX_TYPE = 'box'
CUBE_URDFS = {MESH_TYPE: CUBE_MESH_URDF_ASSET, BOX_TYPE: CUBE_BOX_URDF_ASSET}
ELBOW_URDFS = {MESH_TYPE: ELBOW_MESH_URDF_ASSET, BOX_TYPE: ELBOW_BOX_URDF_ASSET}
TRUE_URDFS = {CUBE_SYSTEM: CUBE_URDFS, ELBOW_SYSTEM: ELBOW_URDFS}


CUBE_BOX_URDF_ASSET_BAD = 'contactnets_cube_bad_init.urdf'
CUBE_BOX_URDF_ASSET_SMALL = 'contactnets_cube_small_init.urdf'
ELBOW_BOX_URDF_ASSET_BAD = 'contactnets_elbow_bad_init.urdf'
ELBOW_BOX_URDF_ASSET_SMALL = 'contactnets_elbow_small_init.urdf'
CUBE_WRONG_URDFS = {'bad': CUBE_BOX_URDF_ASSET_BAD,
                    'small': CUBE_BOX_URDF_ASSET_SMALL}
ELBOW_WRONG_URDFS = {'bad': ELBOW_BOX_URDF_ASSET_BAD,
                    'small': ELBOW_BOX_URDF_ASSET_SMALL}
WRONG_URDFS = {CUBE_SYSTEM: CUBE_WRONG_URDFS, ELBOW_SYSTEM: ELBOW_WRONG_URDFS}


REPO_DIR = os.path.normpath(
    git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel"))

# Data configuration.
DT = 0.0068

# Generation configuration.
CUBE_X_0 = torch.tensor(
    [1., 0., 0., 0., 0., 0., 0.21 + .015, 0., 0., 0., 0., 0., -.075])
ELBOW_X_0 = torch.tensor(
    [1., 0., 0., 0., 0., 0., 0.21 + .015, np.pi, 0., 0., 0., 0., 0., -.075, 0.])
X_0S = {CUBE_SYSTEM: CUBE_X_0, ELBOW_SYSTEM: ELBOW_X_0}
CUBE_SAMPLER_RANGE = torch.tensor([
    2 * np.pi, 2 * np.pi, 2 * np.pi, .03, .03, .015, 6., 6., 6., 1.5, 1.5, .075
])
ELBOW_SAMPLER_RANGE = torch.tensor([
    2 * np.pi, 2 * np.pi, 2 * np.pi, .03, .03, .015, np.pi, 6., 6., 6., 1.5,
    1.5, .075, 6.
])
SAMPLER_RANGES = {
    CUBE_SYSTEM: CUBE_SAMPLER_RANGE,
    ELBOW_SYSTEM: ELBOW_SAMPLER_RANGE
}
TRAJECTORY_LENGTHS = {CUBE_SYSTEM: 80, ELBOW_SYSTEM: 120}

# Training data configuration.
T_PREDICTION = 1

# Optimization configuration.
CUBE_LR = 1e-3
ELBOW_LR = 1e-3
LRS = {CUBE_SYSTEM: CUBE_LR, ELBOW_SYSTEM: ELBOW_LR}
CUBE_WD = 0.0
ELBOW_WD = 0.0  #1e-4
WDS = {CUBE_SYSTEM: CUBE_WD, ELBOW_SYSTEM: ELBOW_WD}
EPOCHS = 500            # change this (originally 500)
PATIENCE = 200       # change this (originally EPOCHS)
DEFAULT_LOSS_WEIGHT_RANGE = (1e-2, 1e2)

WANDB_DEFAULT_PROJECT = 'dair_pll-examples'


def main(storage_folder_name: str = "",
         run_name: str = "",
         system: str = CUBE_SYSTEM,
         source: str = SIM_SOURCE,
         contactnets: bool = True,
         box: bool = True,
         regenerate: bool = False,
         dataset_size: int = 512,
         inertia_params: str = '4',
         loss_variation: str = '0',
         true_sys: bool = False,
         wandb_project: str = WANDB_DEFAULT_PROJECT,
         w_pred: float = 1e0,
         w_comp: float = 1e0,
         w_diss: float = 1e0,
         w_pen: float = 1e0):
    """Execute ContactNets basic example on a system.

    Args:
        storage_folder_name: name of outer storage directory.
        run_name: name of experiment run.
        system: Which system to learn.
        source: Where to get data from.
        contactnets: Whether to use ContactNets or prediction loss.
        box: Whether to represent geometry as box or mesh.
        regenerate: Whether save updated URDF's each epoch.
        dataset_size: Number of trajectories for train/val/test.
        inertia_params: What inertial parameters to learn.
        true_sys: Whether to start with the "true" URDF or poor initialization.
        wandb_project: What W&B project to store results under.
        w_pred: Weight of prediction term in ContactNets loss.
        w_comp: Weight of complementarity term in ContactNets loss.
        w_diss: Weight of dissipation term in ContactNets loss.
        w_pen: Weight of penetration term in ContactNets loss.
    """
    # pylint: disable=too-many-locals, too-many-arguments

    print(f'Starting test under \'{storage_folder_name}\' ' \
         + f'with name \'{run_name}\':' \
         + f'\n\tPerforming on system: {system} \n\twith source: {source}' \
         + f'\n\tusing ContactNets: {contactnets}' \
         + f'\n\twith box: {box}' \
         + f'\n\tregenerate: {regenerate}' \
         + f'\n\tinertia learning mode: {inertia_params}' \
         + f'\n\twith description: {INERTIA_PARAM_OPTIONS[int(inertia_params)]}' \
         + f'\n\tloss variation: {loss_variation}' \
         + f'\n\twith description: {LOSS_VARIATIONS[int(loss_variation)]}' \
         + f'\n\tloss weights (w_pred, w_comp, w_diss, w_pen): ' \
         + f'({w_pred}, {w_comp}, {w_diss}, {w_pen})' \
         + f'\n\tand starting with "true" URDF: {true_sys}.')

    simulation = source == SIM_SOURCE
    dynamic = source == DYNAMIC_SOURCE

    storage_name = os.path.join(REPO_DIR, 'results', storage_folder_name)

    # If this script is used in conjuction with pll_manager.py, then the file
    # management is taken care of there.

    print(f'\nStoring data at    {file_utils.data_dir(storage_name)}')
    print(f'Storing results at {file_utils.run_dir(storage_name, run_name)}')

    # Next, build the configuration of the learning experiment.

    # Describes the optimizer settings; by default, the optimizer is Adam.
    optimizer_config = OptimizerConfig(lr=Float(LRS[system]),
                                       wd=Float(WDS[system]),
                                       patience=PATIENCE,
                                       epochs=EPOCHS,
                                       batch_size=Int(int(dataset_size/2)))

    # Describes the ground truth system; infers everything from the URDF.
    # This is a configuration for a DrakeSystem, which wraps a Drake
    # simulation for the described URDFs.
    # first, select urdfs
    urdf_asset = TRUE_URDFS[system][BOX_TYPE if box else MESH_TYPE]
    urdf = file_utils.get_asset(urdf_asset)
    urdfs = {system: urdf}
    base_config = DrakeSystemConfig(urdfs=urdfs)

    # Describes the learnable system. The MultibodyLearnableSystem type learns
    # a multibody system, which is initialized as the original system URDF, or
    # as a provided wrong initialization. For now, this is only implemented with
    # the box geoemtry parameterization.
    if box and not true_sys:
        wrong_urdf_asset = WRONG_URDFS[system]['small']
        wrong_urdf = file_utils.get_asset(wrong_urdf_asset)
        init_urdfs = {system: wrong_urdf}
    # else:  use the initial mesh type anyway
    else:
        init_urdfs = urdfs

    loss = MultibodyLosses.CONTACTNETS_LOSS \
        if contactnets else \
        MultibodyLosses.PREDICTION_LOSS

    learnable_config = MultibodyLearnableSystemConfig(
        urdfs=init_urdfs, loss=loss, inertia_mode=int(inertia_params),
        loss_variation=int(loss_variation), w_pred=w_pred,
        w_comp=Float(w_comp, log=True, distribution=DEFAULT_LOSS_WEIGHT_RANGE),
        w_diss=Float(w_diss, log=True, distribution=DEFAULT_LOSS_WEIGHT_RANGE),
        w_pen=Float(w_pen, log=True, distribution=DEFAULT_LOSS_WEIGHT_RANGE))

    # how to slice trajectories into training datapoints
    slice_config = TrajectorySliceConfig(
        t_prediction=1 if contactnets else T_PREDICTION)

    # Describes configuration of the data
    data_config = DataConfig(dt=DT,
                             train_fraction=1.0 if dynamic else 0.5,
                             valid_fraction=0.0 if dynamic else 0.25,
                             test_fraction=0.0 if dynamic else 0.25,
                             slice_config=slice_config,
                             update_dynamically=dynamic)

    # Combines everything into config for entire experiment.
    experiment_config = DrakeMultibodyLearnableExperimentConfig(
        storage=storage_name,
        run_name=run_name,
        base_config=base_config,
        learnable_config=learnable_config,
        optimizer_config=optimizer_config,
        data_config=data_config,
        full_evaluation_period=EPOCHS if dynamic else 1,
        # full_evaluation_samples=dataset_size,  # use all data for eval
        visualize_learned_geometry=True,
        run_wandb=True,
        wandb_project=wandb_project
    )

    # Makes experiment.
    experiment = DrakeMultibodyLearnableExperiment(experiment_config)

    # Prepare data.
    x_0 = X_0S[system]
    if simulation:

        # For simulation, specify the following:
        data_generation_config = DataGenerationConfig(
            dt=DT,
            # timestep
            n_pop=dataset_size,
            # How many trajectories to simulate
            trajectory_length=TRAJECTORY_LENGTHS[system],
            # trajectory length
            x_0=x_0,
            # A nominal initial state
            sampler_type=UniformSampler,
            # use uniform distribution to sample ``x_0``
            sampler_ranges=SAMPLER_RANGES[system],
            # How much to vary initial states around ``x_0``
            noiser_type=GaussianWhiteNoiser,
            # Distribution of noise in trajectory data (Gaussian).
            static_noise=torch.zeros(x_0.nelement() - 1),
            # constant-in-time noise standard deviations (zero in this case)
            dynamic_noise=torch.zeros(x_0.nelement() - 1),
            # i.i.d.-in-time noise standard deviations (zero in this case)
            storage=storage_name
            # where to store trajectories
        )

        generator = ExperimentDatasetGenerator(experiment.get_base_system(),
                                               data_generation_config)
        generator.generate()

    else:
        # otherwise, specify directory with [T, n_x] tensor files saved as
        # 0.pt, 1.pt, ...
        # See :mod:`dair_pll.state_space` for state format.
        data_asset = TRUE_DATA_ASSETS[system]
        import_directory = file_utils.get_asset(data_asset)
        print(f'Getting real trajectories from {import_directory}\n')
        file_utils.import_data_to_storage(storage_name,
                                          import_data_dir=import_directory,
                                          num=dataset_size)

    def regenerate_callback(epoch: int, learned_system: System,
                            train_loss: Tensor,
                            best_valid_loss: Tensor) -> None:
        default_epoch_callback(epoch, learned_system, train_loss,
                               best_valid_loss)
        cast(MultibodyLearnableSystem, learned_system).generate_updated_urdfs()

    # Trains system and saves final results.
    print(f'\nTraining the model.')
    learned_system, stats = experiment.generate_results(
        regenerate_callback if regenerate else default_epoch_callback)

    # Save the final urdf.
    print(f'\nSaving the final learned URDF.')
    learned_system = cast(MultibodyLearnableSystem, learned_system)
    learned_system.generate_updated_urdfs()
    print(f'Done!')




@click.command()
@click.argument('storage_folder_name')
@click.argument('run_name')
@click.option('--system',
              type=click.Choice(SYSTEMS, case_sensitive=True),
              default=CUBE_SYSTEM)
@click.option('--source',
              type=click.Choice(DATA_SOURCES, case_sensitive=True),
              default=SIM_SOURCE)
@click.option('--contactnets/--prediction',
              default=True,
              help="whether to train on ContactNets or prediction loss.")
@click.option('--box/--mesh',
              default=True,
              help="whether to represent geometry as box or mesh.")
@click.option('--regenerate/--no-regenerate',
              default=False,
              help="whether to save updated URDF's each epoch or not.")
@click.option('--dataset-size',
              default=512,
              help="dataset size")
@click.option('--inertia-params',
              type=click.Choice(INERTIA_PARAM_CHOICES),
              default='4',
              help="what inertia parameters to learn.")
@click.option('--loss-variation',
              type=click.Choice(LOSS_VARIATION_NUMBERS),
              default='0',
              help="ContactNets loss variation")
@click.option('--true-sys/--wrong-sys',
              default=False,
              help="whether to start with correct or poor URDF.")
@click.option('--wandb-project',
              type = str,
              default=WANDB_DEFAULT_PROJECT,
              help="what W&B project to save results under.")
@click.option('--w-pred',
              type=float,
              default=1e0,
              help="weight of prediction term in ContactNets loss")
@click.option('--w-comp',
              type=float,
              default=1e0,
              help="weight of complementarity term in ContactNets loss")
@click.option('--w-diss',
              type=float,
              default=1e0,
              help="weight of dissipation term in ContactNets loss")
@click.option('--w-pen',
              type=float,
              default=1e0,
              help="weight of penetration term in ContactNets loss")
def main_command(storage_folder_name: str, run_name: str, system: str,
                 source: str, contactnets: bool, box: bool, regenerate: bool,
                 dataset_size: int, inertia_params: str, loss_variation: str,
                 true_sys: bool, wandb_project: str, w_pred: float,
                 w_comp: float, w_diss: float, w_pen: float):
    """Executes main function with argument interface."""
    assert storage_folder_name is not None
    assert run_name is not None

    main(storage_folder_name, run_name, system, source, contactnets, box,
         regenerate, dataset_size, inertia_params, loss_variation, true_sys,
         wandb_project, w_pred, w_comp, w_diss, w_pen)


if __name__ == '__main__':
    main_command()  # pylint: disable=no-value-for-parameter
