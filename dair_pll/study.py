import copy
import logging
import os.path
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type, Dict, Any, Union, List, Optional

import optuna
import optuna.logging
from torch import Tensor

from dair_pll import file_utils, hyperparameter
from dair_pll.experiment import SupervisedLearningExperiment
from dair_pll.experiment_config import SupervisedLearningExperimentConfig
from dair_pll.hyperparameter import ValueDict
from dair_pll.system import System

OPTUNA_ENVIRONMENT_VARIABLE = 'OPTUNA_SERVER'

OPTUNA_TRIAL_FINISHED_STATES = [
    optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED
]


@dataclass
class OptimalSweepStudyConfig:
    experiment_type: Type[SupervisedLearningExperiment]
    default_experiment_config: SupervisedLearningExperimentConfig
    sweep_domain: Union[List[float], List[int]]
    hyperparameter_dataset_mask: Tensor
    n_trials: int = 100
    min_resource: int = 5
    use_remote_storage: bool = True
    study_name: str = 'default_study_name'

    def __post_init__(self) -> None:
        # default config must use raw dataset size for hyperparameter
        # optimization.
        data_config = self.default_experiment_config.data_config
        assert isinstance(data_config.train_valid_test_quantities[0], int)


class OptimalSweepStudy(ABC):

    config: OptimalSweepStudyConfig

    def __init__(self, config: OptimalSweepStudyConfig) -> None:
        self.config = config

    @abstractmethod
    def set_sweep_value(self,
                        experiment_config: SupervisedLearningExperimentConfig,
                        sweep_value: Any) -> None:
        pass

    def setup_config(self, hyperparameters: ValueDict,
                     run_name: str,
                     is_hyperparameter_optimization: bool,
                     sweep_value: Optional[Union[float, int]] = None) -> \
            SupervisedLearningExperimentConfig:

        experiment_config = copy.deepcopy(self.config.default_experiment_config)
        hyperparameter.load_suggestion(experiment_config, hyperparameters)
        experiment_config.run_name = run_name

        if is_hyperparameter_optimization:
            exclude_mask = ~self.config.hyperparameter_dataset_mask
        else:
            exclude_mask = self.config.hyperparameter_dataset_mask

        experiment_config.data_config.exclude_mask = exclude_mask

        if sweep_value is not None:
            self.set_sweep_value(experiment_config, sweep_value)

        return experiment_config

    def optuna_optimize(self, trial: optuna.trial.Trial) -> float:

        config = self.config
        experiment_suggestion = hyperparameter.generate_suggestion(
            config.default_experiment_config, trial)
        run_name = file_utils.hyperparameter_opt_run_name(
            config.study_name, trial.number)

        trial_experiment_config = self.setup_config(experiment_suggestion,
                                                    run_name, True)

        experiment = config.experiment_type(trial_experiment_config)

        def epoch_callback(epoch: int, _system: System, _train_loss: Tensor,
                           best_valid_loss: Tensor) -> None:
            trial.report(best_valid_loss.item(), step=epoch)
            if trial.should_prune():
                if experiment.config.run_wandb:
                    if experiment.wandb_manager is not None:
                        experiment.wandb_manager.finish()
                raise optuna.TrialPruned()

        _, final_valid_loss, _ = experiment.train(epoch_callback)
        return final_valid_loss.item()

    def run_sweep(self, sweep_number: int) -> None:
        config = self.config

        try:
            hyperparameters = file_utils.load_hyperparameters(
                config.default_experiment_config.storage, config.study_name)
        except FileNotFoundError:
            # N.B. we save and reload the hyperparameters, because some
            # values may change slightly due to floating point precision
            # interacting with JSON serialization.
            self.optimize_hyperparameters()
            hyperparameters = file_utils.load_hyperparameters(
                config.default_experiment_config.storage, config.study_name)

        for sweep_value in config.sweep_domain:
            print(f"running sweep {sweep_number} for value = {sweep_value}")
            sys.stdout.flush()
            self.run_sweep_sample(hyperparameters, sweep_number, sweep_value)
        print("done!")
        sys.stdout.flush()

    def run_sweep_sample(self, hyperparameters: ValueDict, sweep_number: int,
                         sweep_value: Union[float, int]) -> None:
        run_name = file_utils.sweep_run_name(self.config.study_name,
                                             sweep_number, sweep_value)

        sample_experiment_config = self.setup_config(hyperparameters, run_name,
                                                     False, sweep_value)

        experiment = self.config.experiment_type(sample_experiment_config)

        def epoch_cb(_epoch: int, _model: System, _train_loss: Tensor,
                     _best_valid_loss: Tensor) -> None:
            pass

        experiment.generate_results(epoch_cb)

    def is_complete(self, study: optuna.study.Study) -> bool:
        trials = study.trials
        finished = [
            trial for trial in trials
            if trial.state in OPTUNA_TRIAL_FINISHED_STATES
        ]
        return len(finished) >= self.config.n_trials

    def stop_if_complete(self, study: optuna.study.Study, _trial: Any) -> None:
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
            optuna_study = optuna.create_study(
                direction="minimize",
                pruner=pruner,
                study_name=config.study_name,
                storage=os.environ[OPTUNA_ENVIRONMENT_VARIABLE],
                load_if_exists=True)
        else:
            optuna_study = optuna.create_study(direction="minimize",
                                               pruner=pruner,
                                               study_name=config.study_name)
        if not self.is_complete(optuna_study):
            optuna.logging.get_logger("optuna").addHandler(
                logging.StreamHandler(sys.stdout))
            optuna_study.optimize(self.optuna_optimize,
                                  n_trials=config.n_trials,
                                  callbacks=[self.stop_if_complete])
        print("Hyperparameter optimization completed.")
        print(optuna_study.best_value)
        file_utils.save_hyperparameters(
            self.config.default_experiment_config.storage,
            self.config.study_name, optuna_study.best_params)
        return optuna_study.best_params


class OptimalDatasizeSweepStudy(OptimalSweepStudy):

    def set_sweep_value(self,
                        experiment_config: SupervisedLearningExperimentConfig,
                        sweep_value: Any) -> None:
        """Sets the train, valid, and test sizes to the sweep value.

        The sweep value is the total number of trajectories in the training set.
        The valid and test sets are half the size of the training set.

        Args:
            experiment_config: The experiment config in which to set the
              dataset sizes.
            sweep_value: The training set size.
        """
        experiment_config.data_config.train_valid_test_quantities = (
            sweep_value, sweep_value // 2, sweep_value // 2)
