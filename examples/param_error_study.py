"""Simple ContactNets/differentiable physics learning examples."""
# pylint: disable=E1103
import os
from copy import deepcopy
from itertools import product

import click
import torch

from contactnets_simple import CUBE_SAMPLER_RANGE, CUBE_X_0, DT, \
    CUBE_MESH_URDF_ASSET, CUBE_SYSTEM, PARAMETER_NOISE_LEVEL, TRAJECTORY_LENGTHS
from dair_pll import file_utils
from dair_pll.dataset_generation import DataGenerationConfig, \
    ExperimentDatasetGenerator
from dair_pll.dataset_management import DataConfig, \
    TrajectorySliceConfig
from dair_pll.drake_experiment import \
    DrakeMultibodyLearnableExperiment, DrakeSystemConfig, \
    MultibodyLearnableSystemConfig, MultibodyLosses, \
    DrakeMultibodyLearnableExperimentConfig
from dair_pll.experiment_config import OptimizerConfig
from dair_pll.geometry import MeshRepresentation
from dair_pll.hyperparameter import Float, Int
from dair_pll.state_space import UniformSampler, GaussianWhiteNoiser
from dair_pll.study import OptimalSweepStudyConfig, OptimalDatasizeSweepStudy

STUDY_NAME_PREFIX = 'noiseless_cube_reconstruction'

# File management.
STORAGE = os.path.join(os.path.dirname(__file__), 'storage', STUDY_NAME_PREFIX)

# Cube system configuration.
SYSTEM = CUBE_SYSTEM
CUBE_URDF_ASSET = CUBE_MESH_URDF_ASSET
URDFS = {SYSTEM: file_utils.get_asset(CUBE_URDF_ASSET)}

# Data configuration.
X_0 = CUBE_X_0
TRAJECTORY_LENGTH = TRAJECTORY_LENGTHS[SYSTEM]

# Generation configuration.
# dataset size isn't really something we need to sweep over; we just need to
# set it high enough such that the parameters are identifiable.
N_HYPERPARMETER_OPTIMIZATION_TRAIN = 64
SWEEP_DOMAIN = [N_HYPERPARMETER_OPTIMIZATION_TRAIN]
N_POP = 4 * (N_HYPERPARMETER_OPTIMIZATION_TRAIN + SWEEP_DOMAIN[-1])

# Optimization configuration.
LR = Float(1e-4, (1e-6, 1e-2), log=True)
WD = Float(0.0, (0.0, 0.0), log=False)
EPOCHS = 100
PATIENCE = 5
BATCH_SIZE = Int(32, (1, 256), log=True)

# Study configuration.
N_TRIALS = 100
MIN_RESOURCES = 5

WANDB_PROJECT = 'contactnets-journal-results'

MESH_REPRESENTATIONS = [MeshRepresentation.POLYGON]
LOSSES = [
    MultibodyLosses.CONTACTNETS_ANITESCU_LOSS,
    MultibodyLosses.PREDICTION_LOSS
]

def name_from_mesh_loss(mesh_representation: MeshRepresentation,
                        loss: MultibodyLosses) -> str:
    """Generate a name for a sweep instance."""
    study_name_postfix = f'{mesh_representation.name}_{loss.name}'
    return f'{STUDY_NAME_PREFIX}_{study_name_postfix}'


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
    learnable_config = MultibodyLearnableSystemConfig(
        urdfs=URDFS, initial_parameter_noise_level=PARAMETER_NOISE_LEVEL)

    slice_config = TrajectorySliceConfig()
    train_valid_test_quantities = (N_HYPERPARMETER_OPTIMIZATION_TRAIN,
                                   N_HYPERPARMETER_OPTIMIZATION_TRAIN // 2,
                                   N_HYPERPARMETER_OPTIMIZATION_TRAIN // 2)
    total_hyperparameter_trajectories = sum(train_valid_test_quantities)

    # Select hyperparameter trajectories without removing randomness in the
    # seed for the rest of the runs.
    torch_random_generator_state = torch.random.get_rng_state()
    torch.manual_seed(1298361927836198)
    hyperparameter_trajectories = torch.randperm(
        N_POP)[:total_hyperparameter_trajectories]
    hyperparameter_mask = torch.zeros(N_POP, dtype=torch.bool)
    hyperparameter_mask[hyperparameter_trajectories] = True
    torch.random.set_rng_state(torch_random_generator_state)

    data_config = DataConfig(
        dt=DT,
        train_valid_test_quantities=train_valid_test_quantities,
        slice_config=slice_config,
        use_ground_truth=True)

    experiment_config = DrakeMultibodyLearnableExperimentConfig(
        storage=STORAGE,
        run_name='default_run_name',
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
        experiment_config).get_oracle_system()

    generator = ExperimentDatasetGenerator(generation_system,
                                           data_generation_config)
    generator.generate()

    # Run two dataset size sweep studies: one for prediction loss and another
    # for ContactNets loss.
    for mesh_representation, loss in product(MESH_REPRESENTATIONS, LOSSES):
        default_experiment_config = deepcopy(experiment_config)
        assert isinstance(default_experiment_config.learnable_config,
                          MultibodyLearnableSystemConfig)
        default_experiment_config.training_loss = loss
        default_experiment_config.learnable_config.mesh_representation = (
            mesh_representation)

        # setup study config.
        study_config = OptimalSweepStudyConfig(
            sweep_domain=SWEEP_DOMAIN,
            hyperparameter_dataset_mask=hyperparameter_mask,
            n_trials=N_TRIALS,
            min_resource=MIN_RESOURCES,
            use_remote_storage=True,
            study_name=name_from_mesh_loss(mesh_representation, loss),
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
