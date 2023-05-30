"""Simple ContactNets/differentiable physics learning examples."""
# pylint: disable=E1103
import os
from copy import deepcopy
from typing import Tuple, Optional, cast, List

import click
import torch

from contactnets_simple import CUBE_DATA_ASSET, DT, \
    CUBE_MESH_URDF_ASSET, CUBE_SYSTEM, PARAMETER_NOISE_LEVEL
from dair_pll import file_utils
from dair_pll.dataset_management import DataConfig, \
    TrajectorySliceConfig
from dair_pll.deep_learnable_model import MLP
from dair_pll.deep_learnable_system import DeepLearnableSystemConfig
from dair_pll.drake_experiment import \
    DrakeMultibodyLearnableExperiment, DrakeSystemConfig, \
    MultibodyLearnableSystemConfig, MultibodyLosses, \
    DrakeMultibodyLearnableExperimentConfig, DrakeDeepLearnableExperiment
from dair_pll.experiment_config import OptimizerConfig, \
    SupervisedLearningExperimentConfig
from dair_pll.geometry import MeshRepresentation
from dair_pll.hyperparameter import Float, Int
from dair_pll.integrator import DeltaVelocityIntegrator
from dair_pll.study import OptimalSweepStudyConfig, OptimalDatasizeSweepStudy

STUDY_NAME_PREFIX = 'real_cube_reconstruction'

# File management.
STORAGE = os.path.join(os.path.dirname(__file__), 'storage', STUDY_NAME_PREFIX)

# Cube system configuration.
SYSTEM = CUBE_SYSTEM
CUBE_URDF_ASSET = CUBE_MESH_URDF_ASSET
URDFS = {SYSTEM: file_utils.get_asset(CUBE_URDF_ASSET)}

# Data configuration.
IMPORT_DIRECTORY = file_utils.get_asset(CUBE_DATA_ASSET)

# Optimization configuration.
LR = Float(1e-4, (1e-6, 1e-2), log=True)
WD = Float(1e-6, (1e-10, 1e-2), log=True)
EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = Int(32, (1, 256), log=True)

# Experiment configuration.
LEGNTH_SCALE = DT

# Study configuration.
N_POP = 550
SWEEP_DOMAIN = [8, 16, 32, 64, 128]
N_HYPERPARMETER_OPTIMIZATION_TRAIN = SWEEP_DOMAIN[-1]
N_TRIALS = 100
MIN_RESOURCES = 5
MANUAL_BIAS = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, -9.81 * DT])

WANDB_PROJECT = 'contactnets-journal-results'

StudyParams = Tuple[bool, Optional[MeshRepresentation],
                    Optional[MultibodyLosses], str]
SAM_RECREATION_STUDY_PARAMS = (False, None, None, 'SAM')
DEEP_LEARNED_STUDY_PARAMS = (False, None, None, 'MLP')
DEEP_DIFF_PHYS_STUDY_PARAMS = (True, MeshRepresentation.DEEP_SUPPORT_CONVEX,
                               MultibodyLosses.PREDICTION_LOSS,
                               'DEEP_DIFF_PHYS')
DEEP_CONTACTNETS_STUDY_PARAMS = (True, MeshRepresentation.DEEP_SUPPORT_CONVEX,
                                 MultibodyLosses.CONTACTNETS_ANITESCU_LOSS,
                                 'DEEP_CONTACTNETS')
STUDY_PARAMS = cast(
    List[StudyParams],
    [
        #SAM_RECREATION_STUDY_PARAMS,
        #DEEP_LEARNED_STUDY_PARAMS,
        #DEEP_DIFF_PHYS_STUDY_PARAMS,
        DEEP_CONTACTNETS_STUDY_PARAMS,
    ])


def study_name_from_params(params: StudyParams) -> str:
    return f'{STUDY_NAME_PREFIX}_{params[3]}'


def main(sweep_num: int) -> None:
    """Execute hyperparameter-optimal training dataset size sweep.

    This experiment compares the favorability of naive deep learning and
    differentiable physics (prediction error loss) versus ContactNets for
    learning a cube system on real data.
    """
    # pylint: disable=too-many-locals
    # define default experiment config.
    optimizer_config = OptimizerConfig(lr=LR,
                                       wd=WD,
                                       patience=PATIENCE,
                                       epochs=EPOCHS,
                                       batch_size=BATCH_SIZE)

    base_config = DrakeSystemConfig(urdfs=URDFS)
    multibody_learnable_config = MultibodyLearnableSystemConfig(
        urdfs=URDFS, initial_parameter_noise_level=PARAMETER_NOISE_LEVEL)

    deep_learnable_config = DeepLearnableSystemConfig(
        model_constructor=MLP, integrator_type=DeltaVelocityIntegrator,
        manual_bias=MANUAL_BIAS
    )

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

    # Import data.
    #if not os.path.exists(IMPORT_DIRECTORY):
    file_utils.import_data_to_storage(STORAGE,
                                      import_data_dir=IMPORT_DIRECTORY)

    data_config = DataConfig(
        dt=DT,
        train_valid_test_quantities=train_valid_test_quantities,
        slice_config=slice_config,
        use_ground_truth=True)

    multibody_experiment_config = DrakeMultibodyLearnableExperimentConfig(
        contactnets_length_scale=LEGNTH_SCALE,
        storage=STORAGE,
        run_name='default_run_name',
        base_config=base_config,
        learnable_config=multibody_learnable_config,
        optimizer_config=optimizer_config,
        data_config=data_config,
        full_evaluation_period=1,
        visualize_learned_geometry=True,
        run_wandb=True,
        wandb_project=WANDB_PROJECT,
    )

    deep_experiment_config = SupervisedLearningExperimentConfig(
        storage=STORAGE,
        run_name='default_run_name',
        base_config=base_config,
        learnable_config=deep_learnable_config,
        optimizer_config=optimizer_config,
        data_config=data_config,
        full_evaluation_period=1,
        run_wandb=True,
        wandb_project=WANDB_PROJECT)

    for study_params in STUDY_PARAMS:
        if study_params[0]:
            assert study_params[1] is not None
            assert study_params[2] is not None
            default_experiment_config = deepcopy(multibody_experiment_config)
            assert isinstance(default_experiment_config.learnable_config,
                              MultibodyLearnableSystemConfig)
            default_experiment_config.learnable_config.mesh_representation = (
                study_params[1])
            default_experiment_config.training_loss = study_params[2]
            experiment_type = DrakeMultibodyLearnableExperiment
        else:
            default_experiment_config = deepcopy(deep_experiment_config)
            experiment_type = DrakeDeepLearnableExperiment

        # setup study config.
        study_config = OptimalSweepStudyConfig(
            sweep_domain=SWEEP_DOMAIN,
            hyperparameter_dataset_mask=hyperparameter_mask,
            n_trials=N_TRIALS,
            min_resource=MIN_RESOURCES,
            use_remote_storage=True,
            study_name=study_name_from_params(study_params),
            experiment_type=experiment_type,
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
