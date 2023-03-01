"""Simple ContactNets/differentiable physics learning examples."""
# pylint: disable=E1103
import os

import sys
import pdb

import click
import numpy as np
import torch
from torch import Tensor
import pickle
import git

from dair_pll import file_utils
from dair_pll.dataset_management import DataConfig, DataGenerationConfig
from dair_pll.drake_experiment import DrakeMultibodyLearnableExperiment, \
                                      DrakeSystemConfig, \
                                      MultibodyLearnableSystemConfig, \
                                      MultibodyLosses
from dair_pll.experiment import SupervisedLearningExperimentConfig, \
                                OptimizerConfig, default_epoch_callback
from dair_pll.multibody_learnable_system import MultibodyLearnableSystem
from dair_pll.state_space import UniformSampler


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
INERTIA_PARAM_CHOICES = ['0', '1', '2', '3', '4']
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


REPO_DIR = os.path.normpath(git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel"))

# Data configuration.
DT = 0.0068

# Generation configuration.
# N_POP = 256  <-- replaced with a commandline argument
# CUBE_X_0 = torch.tensor([
#     -0.525, 0.394, -0.296, -0.678, 0.186, 0.026, 0.222, 1.463, -4.854, 9.870,
#     0.014, 1.291, -0.212
# ])
CUBE_X_0 = torch.tensor(
    [1., 0., 0., 0., 0., 0., 0.21 + .015, 0., 0., 0., 0., 0., -.075])
ELBOW_X_0 = torch.tensor(
    [1., 0., 0., 0., 0., 0., 0.21 + .015, np.pi, 0., 0., 0., 0., 0., -.075, 0.])
X_0S = {CUBE_SYSTEM: CUBE_X_0, ELBOW_SYSTEM: ELBOW_X_0}
# CUBE_SAMPLER_RANGE = 0.1 * torch.ones(CUBE_X_0.nelement() - 1)
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
TRAJ_LENS = {CUBE_SYSTEM: 80, ELBOW_SYSTEM: 120}

# dynamic load configuration.
DYNAMIC_UPDATES_FROM = 4

# Training data configuration.
T_PREDICTION = 1

# Optimization configuration.
CUBE_LR = 1e-3
ELBOW_LR = 1e-3
LRS = {CUBE_SYSTEM: CUBE_LR, ELBOW_SYSTEM: ELBOW_LR}
CUBE_WD = 0.0
ELBOW_WD = 1e-4
WDS = {CUBE_SYSTEM: CUBE_WD, ELBOW_SYSTEM: ELBOW_WD}
EPOCHS = 500            # change this (originally 500)
PATIENCE = 200       # change this (originally EPOCHS)
# BATCH_SIZE = 256  <-- updated to scale with commandline argument for dataset_size



def main(name: str = None,
         system: str = CUBE_SYSTEM,
         source: str = SIM_SOURCE,
         contactnets: bool = True,
         box: bool = True,
         regenerate: bool = False,
         dataset_size: int = 512,
         local: bool = True,
         videos: bool = False,
         inertia_params: str = '4',
         true_sys: bool = False,
         tb: bool = True):
    """Execute ContactNets basic example on a system.

    Args:
        system: Which system to learn.
        source: Where to get data from.
        contactnets: Whether to use ContactNets or prediction loss.
        box: Whether to represent geometry as box or mesh.
        regenerate: Whether save updated URDF's each epoch.
        dataset_size: Number of trajectories for train/val/test.
        local: Running locally versus on cluster.
        videos: Generate videos or not.
        inertia_params: What inertial parameters to learn.
        true_sys: Whether to start with the "true" URDF or poor initialization.
        tb: Start up tensorboard webpage or not.
    """
    # pylint: disable=too-many-locals

    print(f'\nStarting test with name \'{name}\':' \
         + f'\n\tPerforming on system: {system} \n\twith source: {source}' \
         + f'\n\tusing ContactNets: {contactnets}' \
         + f'\n\twith box: {box}' \
         + f'\n\tregenerate: {regenerate}' \
         + f'\n\trunning locally: {local}' \
         + f'\n\tdoing videos: {videos}' \
         + f'\n\tand inertia learning mode: {inertia_params}' \
         + f'\n\twith description: {INERTIA_PARAM_OPTIONS[int(inertia_params)]}' \
         + f'\n\tand starting with "true" URDF: {true_sys}.')

    storage_name = os.path.join(REPO_DIR, 'results', name)
    print(f'\nStoring data at {storage_name}')

    batch_size = int(dataset_size/2)

    # First step, clear out data on disk for a fresh start.
    simulation = source == SIM_SOURCE
    real = source == REAL_SOURCE
    dynamic = source == DYNAMIC_SOURCE

    data_asset = TRUE_DATA_ASSETS[system]

    # Next, build the configuration of the learning experiment.

    # Describes the optimizer settings; by default, the optimizer is Adam.
    optimizer_config = OptimizerConfig()
    optimizer_config.lr.value = LRS[system]
    optimizer_config.wd.value = WDS[system]
    optimizer_config.patience = PATIENCE
    optimizer_config.epochs = EPOCHS
    optimizer_config.batch_size.value = batch_size

    # Describes the ground truth system; infers everything from the URDF.
    # This is a configuration for a DrakeSystem, which wraps a Drake
    # simulation for the described URDFs.
    # first, select urdfs
    urdf_asset = TRUE_URDFS[system][BOX_TYPE if box else MESH_TYPE]
    urdf = file_utils.get_asset(urdf_asset)
    urdfs = {system: urdf}
    base_config = DrakeSystemConfig(urdfs=urdfs)

    # Describes the learnable system. The MultibodyLearnableSystem type
    # learns a multibody system, which is initialized as the original
    # system URDF, or as a provided wrong initialization.
    # For now, this is only implemented with the box geometry
    # parameterization.
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
        urdfs=init_urdfs, loss=loss, inertia_mode=int(inertia_params))

    # Describe data source
    data_generation_config = None
    import_directory = None
    dynamic_updates_from = None
    x_0 = X_0S[system]
    if simulation:
        # For simulation, specify the following:
        data_generation_config = DataGenerationConfig(
            n_pop=dataset_size,
            # How many trajectories to simulate
            x_0=x_0,
            # A nominal initial state
            sampler_ranges=SAMPLER_RANGES[system],
            # How much to vary initial states around ``x_0``
            sampler_type=UniformSampler,
            # use uniform distribution to sample ``x_0``
            static_noise=torch.zeros(x_0.nelement() - 1),
            # constant-in-time noise distribution (zero in this case)
            dynamic_noise=torch.zeros(x_0.nelement() - 1),
            # i.i.d.-in-time noise distribution (zero in this case)
            traj_len=TRAJ_LENS[system])
    elif real:
        # otherwise, specify directory with [T, n_x] tensor files saved as
        # 0.pt, 1.pt, ...
        # See :mod:`dair_pll.state_space` for state format.
        import_directory = file_utils.get_asset(data_asset)
        print(f'Getting real trajectories from {import_directory}\n')
    else:
        dynamic_updates_from = DYNAMIC_UPDATES_FROM

    # Describes configuration of the data
    data_config = DataConfig(
        storage=storage_name,
        # where to store data
        dt=DT,
        train_fraction=1.0 if dynamic else 0.5,
        valid_fraction=0.0 if dynamic else 0.25,
        test_fraction=0.0 if dynamic else 0.25,
        generation_config=data_generation_config,
        import_directory=import_directory,
        dynamic_updates_from=dynamic_updates_from,
        t_prediction=1 if contactnets else T_PREDICTION,
        n_import=dataset_size if real else None)

    # Combines everything into config for entire experiment.
    experiment_config = SupervisedLearningExperimentConfig(
        base_config=base_config,
        learnable_config=learnable_config,
        optimizer_config=optimizer_config,
        data_config=data_config,
        full_evaluation_period=EPOCHS if dynamic else 1,
        # full_evaluation_samples=dataset_size,  # use all available data for eval
        run_tensorboard=tb,
        gen_videos=videos
    )

    # Makes experiment.
    experiment = DrakeMultibodyLearnableExperiment(experiment_config)

    def regenerate_callback(epoch: int,
                            learned_system: MultibodyLearnableSystem,
                            train_loss: Tensor,
                            best_valid_loss: Tensor) -> None:
        default_epoch_callback(
            epoch, learned_system, train_loss, best_valid_loss)
        learned_system.generate_updated_urdfs(storage_name)

    def log_callback(epoch: int,
                     learned_system: MultibodyLearnableSystem,
                     train_loss: Tensor,
                     best_valid_loss: Tensor) -> None:
        default_epoch_callback(epoch, learned_system, train_loss,
                               best_valid_loss)

        scalars, _ = learned_system.multibody_terms.scalars_and_meshes()
        stats = {}

        for key in ['train_model_trajectory_mse', 'valid_model_trajectory_mse',
                    'train_model_trajectory_mse_mean',
                    'valid_model_trajectory_mse_mean',
                    'train_delta_v_squared_mean', 'valid_delta_v_squared_mean',
                    'train_v_plus_squared_mean', 'valid_v_plus_squared_mean',
                    'train_model_loss_mean', 'valid_model_loss_mean',
                    'training_duration', 'evaluation_duration',
                    'logging_duration']:
            stats[key] = experiment.statistics[key]

        with open(f'{storage_name}/params.txt', 'a') as txt_file:
            txt_file.write(f'Epoch {epoch}:\n\tscalars: {scalars}\n' \
                           + f'\tstatistics: {stats}\n' \
                           + f'\ttrain_loss: {train_loss}\n\n')
            txt_file.close()

    def log_and_regen_callback(epoch: int,
                     learned_system: MultibodyLearnableSystem,
                     train_loss: Tensor,
                     best_valid_loss: Tensor) -> None:
        log_callback(epoch, learned_system, train_loss, best_valid_loss)
        learned_system.generate_updated_urdfs(storage_name)


    # Save all parameters so far in experiment directory.
    # with open(f'{storage_name}/params.pickle', 'wb') as pickle_file:
    #     pickle.dump(experiment_config, pickle_file)
    # with open(f'{storage_name}/dataset.pickle', 'wb') as pickle_file:
    #     pickle.dump(experiment.data_manager.orig_data, pickle_file)
    with open(f'{storage_name}/params.txt', 'a') as txt_file:
        if source == REAL_SOURCE:
            orig_data = f'experiment_config.data_manager.orig_data:' \
                        + f'{experiment.data_manager.orig_data}\n\n'
        else:
            orig_data = ''

        txt_file.write(f'Starting test with name \'{name}\':' \
            + f'\n\tPerforming on system: {system}\n\twith source: {source}' \
            + f'\n\tusing ContactNets: {contactnets}' \
            + f'\n\twith box: {box}' \
            + f'\n\tregenerate: {regenerate}' \
            + f'\n\trunning locally: {local}' \
            + f'\n\tdoing videos: {videos}' \
            + f'\n\tand inertia learning mode: {inertia_params}' \
            + f'\n\twith description: {INERTIA_PARAM_OPTIONS[int(inertia_params)]}' \
            + f'\n\tand starting with "true" URDF: {true_sys}.' \
            + f'\n\nexperiment_config: {experiment_config}\n\n' \
            + orig_data \
            + f'optimizer_config.lr:  {optimizer_config.lr.value}\n' \
            + f'optimizer_config.wd:  {optimizer_config.wd.value}\n' \
            + f'optimizer_config.batch_size:  {optimizer_config.batch_size.value}\n\n')
        
        train_set, _, _ = experiment.data_manager.get_trajectory_split()
        if hasattr(experiment.data_manager, 'train_indices'):
            train_indices = experiment.data_manager.train_indices
            valid_indices = experiment.data_manager.valid_indices
            test_indices = experiment.data_manager.test_indices
            txt_file.write(f'training set data indices:  {train_indices}\n' \
                + f'validation set data indices:  {valid_indices}\n' \
                + f'test set data indices:  {test_indices}\n\n')

    # Trains system.
    _, _, learned_system = experiment.train(
        log_and_regen_callback if regenerate else log_callback #default_epoch_callback
    )

    # Save the final urdf.
    print(f'\nSaving the final learned box parameters.')
    learned_system.generate_updated_urdfs(storage_name)



@click.command()
@click.argument('name')
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
@click.option('--local/--cluster',
              default=False,
              help="whether running script locally or on cluster.")
@click.option('--videos/--no-videos',
              default=False,
              help="whether to generate videos or not.")
@click.option('--inertia-params',
              type=click.Choice(INERTIA_PARAM_CHOICES),
              default='4',
              help="what inertia parameters to learn.")
@click.option('--true-sys/--wrong-sys',
              default=False,
              help="whether to start with correct or poor URDF.")
@click.option('--tb/--no-tb',
              default=False,
              help="whether to start tensorboard webpage.")
def main_command(name: str, system: str, source: str, contactnets: bool,
                 box: bool, regenerate: bool, dataset_size: int, local: bool,
                 videos: bool, inertia_params: str, true_sys: bool,
                 tb: bool):
    """Executes main function with argument interface."""
    assert name is not None

    main(name, system, source, contactnets, box, regenerate, dataset_size,
         local, videos, inertia_params, true_sys, tb)


if __name__ == '__main__':
    main_command()  # pylint: disable=no-value-for-parameter
