import copy
import logging
import os.path
import sys
from dataclasses import dataclass
from typing import Tuple, Type, Dict, Any

import optuna
import optuna.logging
from torch import Tensor
from torch.nn import Module

from dair_pll import file_utils, hyperparameter
from dair_pll.dataset_management import DataConfig
from dair_pll.drake_experiment import DrakeSystemConfig, \
    MultibodyLearnableSystemConfig, DrakeMultibodyLearnableExperiment
from dair_pll.experiment import SupervisedLearningExperiment
from dair_pll.experiment_config import SupervisedLearningExperimentConfig
from dair_pll.system import System

OPTUNA_ENVIRONMENT_VARIABLE = 'OPTUNA_SERVER'

OPTUNA_TRIAL_FINISHED_STATES = [optuna.trial.TrialState.COMPLETE,
                               optuna.trial.TrialState.PRUNED]


@dataclass
class StudyConfig:
    n_trials: int = 100
    min_resource: int = 5
    n_sweep_runs: int = 5
    log_data_size_range: Tuple[int, int] = (3, 12)
    use_remote_storage: bool = True
    study_name: str = ''
    experiment_type: Type[
        SupervisedLearningExperiment] = SupervisedLearningExperiment
    default_experiment_config: SupervisedLearningExperimentConfig = SupervisedLearningExperimentConfig(
    )


class Study:
    best_params: Dict = {}

    def __init__(self, config: StudyConfig) -> None:
        self.config = config

    def optimize(self, trial: optuna.trial.Trial) -> float:

        def epoch_callback(epoch: int, _system: System, _train_loss: Tensor,
                           best_valid_loss: Tensor) -> None:
            trial.report(best_valid_loss.item(), step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        config = self.config
        experiment_suggestion = hyperparameter.generate_suggestion(
            config.default_experiment_config, trial)

        trial_experiment_config = copy.deepcopy(
            config.default_experiment_config)

        hyperparameter.load_suggestion(trial_experiment_config,
                                       experiment_suggestion)

        run_name = file_utils.hyperparameter_opt_run_name(
            config.study_name, trial.number)

        trial_experiment_config.run_name = run_name

        experiment = config.experiment_type(trial_experiment_config)
        _, best_valid_loss, _ = experiment.train(epoch_callback)
        return best_valid_loss.item()

    def study(self) -> None:
        config = self.config
        log_data_size_range = config.log_data_size_range
        # N_train = config.default_experiment_config.data_config.N_train

        hps = file_utils.load_hyperparameters(
            self.config.default_experiment_config.storage,
            self.config.study_name)

        data_min = log_data_size_range[0]
        data_max = log_data_size_range[1] + 1
        for sweep_run in range(config.n_sweep_runs):
            for log_N_train in range(data_min, data_max):
                N_train = 2**log_N_train
                print(f"running sweep example for N_train = {N_train}")
                sys.stdout.flush()
                self.run_datasweep_sample(hps, sweep_run, N_train)
        print("done!")
        sys.stdout.flush()

    def run_datasweep_sample(self, hps: hyperparameter.ValueDict,
                             sweep_run: int, N_train: int) -> None:
        sample_experiment_config = copy.deepcopy(
            self.config.default_experiment_config)
        hyperparameter.load_suggestion(sample_experiment_config, hps)
        sample_experiment_config.data_config.n_train = N_train

        sample_experiment_config.run_name = file_utils.sweep_run_name(
            self.config.study_name, sweep_run, N_train)

        experiment = self.config.experiment_type(sample_experiment_config)

        def epoch_cb(epoch: int, model: Module, train_loss: float,
                     best_valid_loss: float) -> None:
            pass

        experiment.get_results(epoch_cb)

    def is_complete(self, study: optuna.study.Study) -> bool:
        trials = study.trials
        finished = [
            trial for trial in trials
            if trial.state in OPTUNA_TRIAL_FINISHED_STATES
        ]
        return len(finished) >= self.config.n_trials

    def stop_if_complete(self, study: optuna.study.Study,
                         _: optuna.trial._frozen.FrozenTrial) -> None:
        if self.is_complete(study):
            study.stop()

    def optimize_hyperparameters(self) -> Dict[str, Any]:
        config = self.config
        optimizer_config = config.default_experiment_config.optimizer_config

        pruner = optuna.pruners.HyperbandPruner(
            min_resource=config.min_resource,
            max_resource=optimizer_config.epochs)
        if config.use_remote_storage:
            if not OPTUNA_ENVIRONMENT_VARIABLE in os.environ:
                raise EnvironmentError('Must set '
                                       f'{OPTUNA_ENVIRONMENT_VARIABLE} '
                                       'to optuna server URI!')
            study = optuna.create_study(
                direction="minimize",
                pruner=pruner,
                study_name=config.study_name,
                storage=os.environ[OPTUNA_ENVIRONMENT_VARIABLE],
                load_if_exists=True)
        else:
            study = optuna.create_study(direction="minimize",
                                        pruner=pruner,
                                        study_name=config.study_name)
        if not self.is_complete(study):
            optuna.logging.get_logger("optuna").addHandler(
                logging.StreamHandler(sys.stdout))
            study.optimize(self.optimize,
                           n_trials=config.n_trials,
                           callbacks=[self.stop_if_complete])
        print("Study completed!")
        print(study.best_value)
        file_utils.save_hyperparameters(
            self.config.default_experiment_config.storage,
            self.config.study_name, study.best_params)
        return study.best_params


if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    CUBE_DATA_ASSET = 'contactnets_cube'
    BOX_URDF_ASSET = 'contactnets_cube.urdf'
    CUBE_MODEL = 'cube'
    STUDY_NAME = f'{CUBE_DATA_ASSET}_study'
    STORAGE_NAME = os.path.join(os.path.dirname(__file__), 'storage',
                                STUDY_NAME)
    os.system(f'rm -r {file_utils.storage_dir(STORAGE_NAME)}')

    DT = 1. / 148.

    cube_urdf = file_utils.get_asset(BOX_URDF_ASSET)
    urdfs = {CUBE_MODEL: cube_urdf}
    base_config = DrakeSystemConfig(urdfs=urdfs)
    learnable_config = MultibodyLearnableSystemConfig(urdfs=urdfs)

    import_directory = file_utils.get_asset(CUBE_DATA_ASSET)

    data_config = DataConfig(storage=STORAGE_NAME,
                             dt=DT,
                             n_train=4,
                             n_valid=2,
                             n_test=2,
                             import_directory=import_directory)

    default_experiment_config = SupervisedLearningExperimentConfig(
        base_config=base_config,
        learnable_config=learnable_config,
        data_config=data_config,
    )

    study_config = StudyConfig(
        study_name=STUDY_NAME,
        default_experiment_config=default_experiment_config,
        experiment_type=DrakeMultibodyLearnableExperiment,
        use_remote_storage=False)

    study = Study(study_config)
    study.study()
