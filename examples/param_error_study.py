"""Simple ContactNets/differentiable physics learning examples."""
# pylint: disable=E1103
import os
from copy import deepcopy

import click
import torch
from torch import Tensor

from dair_pll.dataset_generation import DataGenerationConfig, \
    ExperimentDatasetGenerator
from dair_pll.dataset_management import DataConfig, \
    TrajectorySliceConfig
from dair_pll.drake_experiment import \
    DrakeMultibodyLearnableExperiment, DrakeSystemConfig, \
    MultibodyLearnableSystemConfig, MultibodyLosses, \
    DrakeMultibodyLearnableExperimentConfig
from dair_pll.experiment_config import OptimizerConfig
from dair_pll.hyperparameter import Float, Int
from dair_pll.state_space import UniformSampler, GaussianWhiteNoiser
from dair_pll.study import OptimalSweepStudyConfig, OptimalDatasizeSweepStudy

STUDY_NAME = 'noiseless_cube_reconstruction'

# File management.
STORAGE = os.path.join(os.path.dirname(__file__), 'storage', STUDY_NAME)

# Cube system definitions.
SYSTEM = 'cube'
CUBE_DATA_ASSET = 'contactnets_cube'
CUBE_URDF_ASSET = 'contactnets_cube_mesh.urdf'
URDFS = {SYSTEM: CUBE_URDF_ASSET}

# Data configuration.
DT = 0.0068
X_0 = torch.tensor([
    -0.525, 0.394, -0.296, -0.678, 0.186, 0.026, 0.222, 1.463, -4.854, 9.870,
    0.014, 1.291, -0.212
])
CUBE_SAMPLER_RANGE = 0.1 * torch.ones(X_0.nelement() - 1)
TRAJECTORY_LENGTH = 80

# Generation configuration.
N_HYPERPARMETER_OPTIMIZATION_TRAIN = 2**5
SWEEP_DOMAIN = [2**i for i in range(3, 8)]
N_POP = 4 * (N_HYPERPARMETER_OPTIMIZATION_TRAIN + SWEEP_DOMAIN[-1])

# Optimization configuration.
LR = Float(1e-4, (1e-6, 1e-2), log=True)
WD = Float(1e-6, (1e-10, 1e-2), log=True)
EPOCHS = 1000
PATIENCE = 50
BATCH_SIZE = Int(32, (1, 256), log=True)

WANDB_PROJECT = 'contactnets-results'

# TODOS:
# - [ ] Set up initial noise logging.
# - [ ] Select ground-truth data usage.


def main(sweep_num: int) -> None:
    """Execute hyperparameter-optimal training dataset size sweep.

    This experiment compares the favorability of differentiable physics (
    prediction error loss) versus ContactNets for learning a cube system. To
    isolate the performance of the two methods, we use a noiseless dataset,
    and compare training errors before and after training.
    """
    # pylint: disable=too-many-locals
    # define default experiment config.
    optimizer_config = OptimizerConfig(lr=LR,
                                       wd=WD,
                                       patience=PATIENCE,
                                       epochs=EPOCHS,
                                       batch_size=BATCH_SIZE)

    base_config = DrakeSystemConfig(urdfs=URDFS)
    learnable_config = MultibodyLearnableSystemConfig(urdfs=URDFS)

    slice_config = TrajectorySliceConfig()
    train_valid_test_quantities = (N_HYPERPARMETER_OPTIMIZATION_TRAIN,
                                   N_HYPERPARMETER_OPTIMIZATION_TRAIN // 2,
                                   N_HYPERPARMETER_OPTIMIZATION_TRAIN // 2)
    total_hyperparameter_trajectories = sum(train_valid_test_quantities)

    # Select hyperparameter trajectories without removing randomness in the
    # seed for the rest of the runs.
    torch_random_generator_state = torch.random.get_rng_state()
    torch.manual_seed(12983619278361982)
    hyperparameter_trajectories = torch.randperm(
        N_POP)[:total_hyperparameter_trajectories]
    hyperparameter_mask = torch.zeros(N_POP, dtype=torch.bool)
    hyperparameter_mask[hyperparameter_trajectories] = True
    torch.random.set_rng_state(torch_random_generator_state)

    data_config = DataConfig(
        dt=DT,
        train_valid_test_quantities=train_valid_test_quantities,
        slice_config=slice_config)

    experiment_config = DrakeMultibodyLearnableExperimentConfig(
        storage=STORAGE,
        run_name='',
        base_config=base_config,
        learnable_config=learnable_config,
        optimizer_config=optimizer_config,
        data_config=data_config,
        full_evaluation_period=1,
        visualize_learned_geometry=True,
        run_wandb=True,
        wandb_project=WANDB_PROJECT,
    )

    # generate dataset
    data_generation_config = DataGenerationConfig(
        dt=DT,
        n_pop=N_POP,
        trajectory_length=TRAJECTORY_LENGTH,
        x_0=X_0,
        sampler_type=UniformSampler,
        sampler_ranges=CUBE_SAMPLER_RANGE,
        noiser_type=GaussianWhiteNoiser,
        static_noise=torch.zeros(X_0.nelement() - 1),
        dynamic_noise=torch.zeros(X_0.nelement() - 1),
        storage=STORAGE)

    generation_system = DrakeMultibodyLearnableExperiment(
        experiment_config).get_learned_system(Tensor())

    generator = ExperimentDatasetGenerator(generation_system,
                                           data_generation_config)
    generator.generate()

    # Run two dataset size sweep studies: one for prediction loss and another
    # for ContactNets loss.
    for contactnets in [False, True]:
        default_experiment_config = deepcopy(experiment_config)
        if contactnets:
            default_experiment_config.training_loss = \
                MultibodyLosses.CONTACTNETS_ANITESCU_LOSS

        # setup study config.
        study_config = OptimalSweepStudyConfig(
            sweep_domain=SWEEP_DOMAIN,
            hyperparameter_dataset_mask=hyperparameter_mask,
            n_trials=100,
            min_resource=5,
            use_remote_storage=False,
            study_name=STUDY_NAME,
            experiment_type=DrakeMultibodyLearnableExperiment,
            default_experiment_config=default_experiment_config)

        study = OptimalDatasizeSweepStudy(study_config)

        study.run_sweep(sweep_num)


@click.command()
@click.option('--sweep_num', default=1, show_default=True)
def main_command(sweep_num: int) -> None:
    """Executes main function with ``click`` argument interface."""
    main(sweep_num)


if __name__ == '__main__':
    main_command()  # pylint: disable=no-value-for-parameter
