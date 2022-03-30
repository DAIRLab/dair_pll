import os

import click
import torch

from dair_pll import file_utils
from dair_pll.dataset_management import DataConfig, \
    DataGenerationConfig
from dair_pll.drake_experiment import \
    DrakeMultibodyLearnableExperiment, DrakeSystemConfig, \
    MultibodyLearnableSystemConfig, MultibodyLosses
from dair_pll.experiment import SupervisedLearningExperimentConfig, \
    OptimizerConfig

# File management.
CUBE_DATA_ASSET = 'contactnets_cube'
CUBE_URDF_ASSET = 'contactnets_cube.urdf'
CUBE_MODEL = 'cube'

CUBE_URDF = file_utils.get_asset(CUBE_URDF_ASSET)
URDFS = {CUBE_MODEL: CUBE_URDF}

STORAGE_NAME = os.path.join(os.path.dirname(__file__), 'storage',
                            CUBE_DATA_ASSET)

# Data configuration.
DT = 1 / 148.

# Generation configuration.
POP = 64
X_0 = torch.tensor([[
    -0.525, 0.394, -0.296, -0.678, 0.186, 0.026, 0.222, 1.463, -4.854, 9.870,
    0.014, 1.291, -0.212
]])
SAMPLER_RANGE = 0.1

# Optimization configuration.
LR = 1e-6
WD = 0
PATIENCE = 100
EPOCHS = 100


def main(simulation: bool = True, contactnets: bool = True):
    """Execute ContactNets basic example on the cube system.

    Args:
        simulation: Whether to use simulation or real data.
        contactnets: Whether to use ContactNets or prediction loss
    """
    # First step, clear out data on disc for a fresh start.
    os.system(f'rm -r {file_utils.storage_dir(STORAGE_NAME)}')

    # Next, build the configuration of the learning experiment.

    # Describes the optimizer settings; by default, the optimizer is Adam.
    optimizer_config = OptimizerConfig(lr=LR,
                                       wd=WD,
                                       patience=PATIENCE,
                                       epochs=EPOCHS)

    # Describes the ground truth system; infers everything from the URDF.
    # This is a configuration for a DrakeSystem, which wraps a Drake
    # simulation for the described URDFs.
    base_config = DrakeSystemConfig(urdfs=URDFS)

    # Describes the learnable system. The MultibodyLearnableSystem type
    # learns a multibody system, which is initialized as the system in the
    # given URDFs.
    loss = MultibodyLosses.CONTACTNETS_LOSS \
        if contactnets else \
        MultibodyLosses.PREDICTION_LOSS
    learnable_config = MultibodyLearnableSystemConfig(urdfs=URDFS, loss=loss)

    # Describe data source
    data_generation_config = None
    import_directory = None
    if simulation:
        # For simuylation, specify the following:
        data_generation_config = DataGenerationConfig(
            n_pop=POP,
            # How many trajectories to simulate
            x_0=X_0,
            # A nominal initial state
            sampler_ranges=SAMPLER_RANGE * torch.ones(X_0.nelement() - 1),
            # How much to vary initial states around ``x_0``
            static_noise=torch.zeros(X_0.nelement() - 1),
            # constant-in-time noise distribution (zero in this case)
            dynamic_noise=torch.zeros(X_0.nelement() - 1),
            # i.i.d.-in-time noise distribution (zero in this case)
        )
    else:
        # otherwise, specify directory with [T, n_x] tensor files saved as
        # 0.pt, 1.pt, ...
        # See :mod:`dair_pll.state_space` for state format.
        import_directory = file_utils.get_asset(CUBE_DATA_ASSET)

    # Describes configuration of the data:
    data_config = DataConfig(storage=STORAGE_NAME,
                             # where to store data
                             dt=DT,
                             n_train=POP // 2,
                             n_valid=POP // 4,
                             n_test=POP // 4,
                             generation_config=data_generation_config,
                             import_directory=import_directory)

    # Combines everything into config for entire experiment.
    experiment_config = SupervisedLearningExperimentConfig(
        base_config=base_config,
        learnable_config=learnable_config,
        optimizer_config=optimizer_config,
        data_config=data_config,
    )

    # Makes experiment.
    experiment = DrakeMultibodyLearnableExperiment(experiment_config)

    # Trains system.
    experiment.train()


@click.group()
def cli():
    pass


@cli.command()
@click.option('--simulation/--real',
              default=True,
              help="whether to train/test on simulated or real data.")
@click.option('--contactnets/--prediction',
              default=True,
              help="whether to train/test with ContactNets/prediction loss.")
def main_command(simulation: bool, contactnets: bool):
    main(simulation, contactnets)


if __name__ == '__main__':
    main_command()
