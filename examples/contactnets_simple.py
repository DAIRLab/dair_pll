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
from dair_pll.drake_experiment import DrakeMultibodyLearnableExperiment, DrakeSystemConfig, \
                                      MultibodyLearnableSystemConfig, MultibodyLosses
from dair_pll.experiment import SupervisedLearningExperimentConfig, OptimizerConfig, \
                                default_epoch_callback
from dair_pll.multibody_learnable_system import MultibodyLearnableSystem
from dair_pll.state_space import UniformSampler


CUBE_SYSTEM = 'cube'
ELBOW_SYSTEM = 'elbow'
SYSTEMS = [CUBE_SYSTEM, ELBOW_SYSTEM]
SIM_SOURCE = 'simulation'
REAL_SOURCE = 'real'
DYNAMIC_SOURCE = 'dynamic'
DATA_SOURCES = [SIM_SOURCE, REAL_SOURCE, DYNAMIC_SOURCE]


# File management.
CUBE_DATA_ASSET = 'contactnets_cube'
ELBOW_DATA_ASSET = 'contactnets_elbow'
# CUBE_BOX_URDF_ASSET = 'contactnets_cube_bad_init.urdf'
CUBE_BOX_URDF_ASSET = 'contactnets_cube_small_init.urdf'
CUBE_MESH_URDF_ASSET = 'contactnets_cube_mesh.urdf'
ELBOW_BOX_URDF_ASSET = 'contactnets_elbow.urdf'
ELBOW_MESH_URDF_ASSET = 'contactnets_elbow_mesh.urdf'


DATA_ASSETS = {CUBE_SYSTEM: CUBE_DATA_ASSET, ELBOW_SYSTEM: ELBOW_DATA_ASSET}

MESH_TYPE = 'mesh'
BOX_TYPE = 'box'
CUBE_URDFS = {MESH_TYPE: CUBE_MESH_URDF_ASSET, BOX_TYPE: CUBE_BOX_URDF_ASSET}
ELBOW_URDFS = {MESH_TYPE: ELBOW_MESH_URDF_ASSET, BOX_TYPE: ELBOW_BOX_URDF_ASSET}
URDFS = {CUBE_SYSTEM: CUBE_URDFS, ELBOW_SYSTEM: ELBOW_URDFS}

REPO_DIR = os.path.normpath(git.Repo(search_parent_directories=True).git.rev_parse("--show-toplevel"))

# Data configuration.
DT = 0.0068

# Generation configuration.
# N_POP = 256  <-- replaced with a commandline argument
CUBE_X_0 = torch.tensor([
    -0.525, 0.394, -0.296, -0.678, 0.186, 0.026, 0.222, 1.463, -4.854, 9.870,
    0.014, 1.291, -0.212
])
ELBOW_X_0 = torch.tensor(
    [1., 0., 0., 0., 0., 0., 0.21 + .015, np.pi, 0., 0., 0., 0., 0., -.075, 0.])
X_0S = {CUBE_SYSTEM: CUBE_X_0, ELBOW_SYSTEM: ELBOW_X_0}
CUBE_SAMPLER_RANGE = 0.1 * torch.ones(CUBE_X_0.nelement() - 1)
ELBOW_SAMPLER_RANGE = torch.tensor([
    2 * np.pi, 2 * np.pi, 2 * np.pi, .03, .03, .015, np.pi, 6., 6., 6., .5, .5,
    .075, 6.
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
EPOCHS = 80            # change this (originally 500)
PATIENCE = 20       # change this (originally EPOCHS)
# BATCH_SIZE = 256  <-- updated to scale with commandline argument for dataset_size



def main(name: str = None,
         system: str = CUBE_SYSTEM,
         source: str = SIM_SOURCE,
         contactnets: bool = True,
         box: bool = True,
         regenerate: bool = False,
         dataset_size: int = 512,
         local: bool = True):
    """Execute ContactNets basic example on a system.

    Args:
        system: Which system to learn.
        source: Where to get data from.
        contactnets: Whether to use ContactNets or prediction loss
        box: Whether to represent geometry as box or mesh.
        regenerate: Whether save updated URDF's each epoch.
        dataset_size: Number of trajectories for train/val/test.
        local: Running locally versus on cluster.
    """
    # pylint: disable=too-many-locals

    print(f'\nStarting test with name \'{name}\':' \
         + f'\n\tPerforming on system: {system} \n\twith source: {source}' \
         + f'\n\tusing ContactNets: {contactnets}' \
         + f'\n\twith box: {box} \n\tand regenerate: {regenerate}.')

    # overwrite previous results, per user input.
    storage_name = os.path.join(REPO_DIR, 'results', name)
    os.system(f'rm -r {file_utils.storage_dir(storage_name)}')
    print(f'\nStoring data at {storage_name}')

    BATCH_SIZE = int(dataset_size/2)

    # First step, clear out data on disk for a fresh start.
    simulation = source == SIM_SOURCE
    real = source == REAL_SOURCE
    dynamic = source == DYNAMIC_SOURCE

    data_asset = DATA_ASSETS[system]

    # Next, build the configuration of the learning experiment.

    # Describes the optimizer settings; by default, the optimizer is Adam.
    optimizer_config = OptimizerConfig()
    optimizer_config.lr.value = LRS[system]
    optimizer_config.wd.value = WDS[system]
    optimizer_config.patience = PATIENCE
    optimizer_config.epochs = EPOCHS
    optimizer_config.batch_size.value = BATCH_SIZE

    # Describes the ground truth system; infers everything from the URDF.
    # This is a configuration for a DrakeSystem, which wraps a Drake
    # simulation for the described URDFs.
    # first, select urdfs
    urdf_asset = URDFS[system][BOX_TYPE if box else MESH_TYPE]
    urdf = file_utils.get_asset(urdf_asset)
    urdfs = {system: urdf}
    base_config = DrakeSystemConfig(urdfs=urdfs)

    # Describes the learnable system. The MultibodyLearnableSystem type
    # learns a multibody system, which is initialized as the system in the
    # given URDFs.
    loss = MultibodyLosses.CONTACTNETS_LOSS \
        if contactnets else \
        MultibodyLosses.PREDICTION_LOSS
    learnable_config = MultibodyLearnableSystemConfig(urdfs=urdfs, loss=loss)

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
        full_evaluation_period=EPOCHS if dynamic else 1
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
        default_epoch_callback(epoch, learned_system, train_loss, best_valid_loss)

        scalars, _ = learned_system.multibody_terms.scalars_and_meshes()
        stats = {}
        for key in ['train_model_trajectory_mse', 'valid_model_trajectory_mse',
                    'train_model_trajectory_mse_mean', 'valid_model_trajectory_mse_mean',
                    'training_duration', 'evaluation_duration', 'logging_duration']:
            stats[key] = experiment.statistics[key]

        with open(f'{storage_name}/params.txt', 'a') as txt_file:
            txt_file.write(f'Epoch {epoch}:\n\tscalars: {scalars}\n' \
                           + f'\tstatistics: {stats}\n' \
                           + f'\ttrain_loss: {train_loss}\n\n')
            txt_file.close()

    # Save all parameters so far in experiment directory.
    # with open(f'{storage_name}/params.pickle', 'wb') as pickle_file:
    #     pickle.dump(experiment_config, pickle_file)
    # with open(f'{storage_name}/dataset.pickle', 'wb') as pickle_file:
    #     pickle.dump(experiment.data_manager.orig_data, pickle_file)
    with open(f'{storage_name}/params.txt', 'a') as txt_file:
        txt_file.write(f'Starting test with name \'{name}\':' \
            + f'\n\tPerforming on system: {system} \n\twith source: {source}' \
            + f'\n\tusing ContactNets: {contactnets}' \
            + f'\n\twith box: {box} \n\tand regenerate: {regenerate}.\n\n' \
            + f'experiment_config: {experiment_config}\n\n' \
            + f'experiment_config.data_manager.orig_data:' \
            + f'{experiment.data_manager.orig_data}\n\n' \
            + f'optimizer_config.lr:  {optimizer_config.lr.value}\n\n' \
            + f'optimizer_config.wd:  {optimizer_config.wd.value}\n\n' \
            + f'optimizer_config.batch_size:  {optimizer_config.batch_size.value}\n\n')
        
        train_set, _, _ = experiment.data_manager.get_trajectory_split()
        learned_system = experiment.get_learned_system(torch.cat(train_set.trajectories))
        scalars, _ = learned_system.multibody_terms.scalars_and_meshes()
        txt_file.write(f'Epoch 0:\n\tscalars: {scalars}\n\n')
        txt_file.close()

    # Trains system.
    experiment.train(
        regenerate_callback if regenerate else log_callback #default_epoch_callback
    )

    # Save the final urdf.
    print(f'\nSaving the final learned box parameters.')
    train_set, _, _ = experiment.data_manager.get_trajectory_split()
    learned_system = experiment.get_learned_system(torch.cat(train_set.trajectories))
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
              help="whether to train/test with ContactNets/prediction loss.")
@click.option('--box/--mesh',
              default=True,
              help="whether to represent geometry as box or mesh.")
@click.option('--regenerate/--no-regenerate',
              default=False,
              help="whether save updated URDF's each epoch.")
@click.option('--dataset-size',
              default=512,
              help="dataset size")
@click.option('--local/--cluster',
              default=True,
              help="running script locally or on cluster.")
def main_command(name: str, system: str, source: str, contactnets: bool,
                 box: bool, regenerate: bool, dataset_size: int, local: bool):
    """Executes main function with argument interface."""
    assert name is not None

    main(name, system, source, contactnets, box, regenerate, dataset_size, local)


if __name__ == '__main__':
    main_command()  # pylint: disable=no-value-for-parameter