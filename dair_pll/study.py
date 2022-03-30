import copy
import json
import logging
import os.path
import pickle
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Type, Dict

import optuna
import optuna.logging
import torch
from torch.nn import Module

from dair_pll import file_utils
from dair_pll import integrator
from dair_pll import model
from dair_pll.experiment import SupervisedLearningExperiment, \
    MuJoCoDataExperiment
from dair_pll.experiment import SupervisedLearningExperimentConfig, \
    MuJoCoDataExperimentConfig


@dataclass
class StudyConfig:
    lr_range: Tuple[float, float] = (1e-5, 1e-2)
    wd_range: Tuple[float, float] = (1e-8, 1e-2)
    log_batch_size: Tuple[int, int] = (0, 8)
    integrator_types: List[str] = field(
        default_factory=lambda: ['VelocityIntegrator',
                                 'DeltaVelocityIntegrator'])
    model_types: List[str] = field(
        default_factory=lambda: ['RecurrentModel', 'MLP'])
    nonlinearities: List[str] = field(default_factory=lambda: ['ReLU', 'Tanh'])
    log_history_range: Tuple[int, int] = (0, 4)
    log_hidden_size_range: Tuple[int, int] = (6, 9)
    layers_range: Tuple[int, int] = (1, 8)
    n_trials: int = 100
    min_resource: int = 5

    log_data_size_range: Tuple[int] = (3, 12)

    remote_storage: bool = True
    study_name: str = ''
    experiment_type: Type[
        SupervisedLearningExperiment] = SupervisedLearningExperiment
    default_experiment_config: SupervisedLearningExperimentConfig = SupervisedLearningExperimentConfig()


class Study():
    best_params: Dict = {}

    def __init__(self, config: StudyConfig) -> None:
        self.config = config

    def load_config_from_suggestion(self,
                                    suggestion: Dict) -> SupervisedLearningExperimentConfig:
        config = self.config
        experiment_config = copy.deepcopy(config.default_experiment_config)

        experiment_config.optimizer_config.lr = suggestion['lr']
        experiment_config.optimizer_config.wd = suggestion['wd']
        experiment_config.optimizer_config.batch_size = 2 ** suggestion[
            'log_batch_size']
        # pdb.set_trace()

        experiment_config.learnable_config.integrator_type = getattr(integrator,
                                                                     suggestion[
                                                                         'integrator'])
        experiment_config.learnable_config.model_constructor = getattr(model,
                                                                       suggestion[
                                                                           'model_constructor'])
        experiment_config.learnable_config.nonlinearity = getattr(torch.nn,
                                                                  suggestion[
                                                                      'nonlinearity'])
        experiment_config.learnable_config.t_history = 2 ** suggestion[
            'log_T_history']
        experiment_config.learnable_config.hidden_size = 2 ** suggestion[
            'log_hidden_size']
        experiment_config.learnable_config.layers = suggestion['layers']

        return experiment_config

    def suggest_config(self,
                       trial: optuna.trial.Trial) -> SupervisedLearningExperimentConfig:
        config = self.config

        lr = trial.suggest_loguniform("lr", *config.lr_range)
        wd = trial.suggest_loguniform("wd", *config.wd_range)
        log_batch_size = trial.suggest_int('log_batch_size',
                                           *config.log_batch_size)

        integrator_type = trial.suggest_categorical('integrator',
                                                    config.integrator_types)
        model_constructor = trial.suggest_categorical('model_constructor',
                                                      config.model_types)
        nonlinearity = trial.suggest_categorical('nonlinearity',
                                                 config.nonlinearities)
        log_T_history = trial.suggest_int('log_T_history',
                                          *config.log_history_range)
        log_hidden_size = trial.suggest_int('log_hidden_size',
                                            *config.log_hidden_size_range)
        layers = trial.suggest_int('layers', *config.layers_range)

        return {
            'lr': lr,
            'wd': wd,
            'log_batch_size': log_batch_size,
            'integrator': integrator_type,
            'model_constructor': model_constructor,
            'nonlinearity': nonlinearity,
            'log_T_history': log_T_history,
            'log_hidden_size': log_hidden_size,
            'layers': layers
        }

    def optimize(self, trial: optuna.trial.Trial) -> None:

        def epoch_cb(epoch: int, model: Module, train_loss: float,
                     best_valid_loss: float) -> None:
            trial.report(best_valid_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        config = self.config
        experiment_suggestion = self.suggest_config(trial)
        trial_experiment_config = self.load_config_from_suggestion(
            experiment_suggestion)
        experiment = config.experiment_type(trial_experiment_config)
        _, best_valid_loss, _ = experiment.train(epoch_cb)
        return best_valid_loss

    def study(self) -> None:
        config = self.config
        log_data_size_range = config.log_data_size_range
        # N_train = config.default_experiment_config.data_config.N_train

        hps = self.get_hpopt()

        data_min = log_data_size_range[0]
        data_max = log_data_size_range[1] + 1
        for log_N_train in range(data_min, data_max):
            N_train = 2 ** log_N_train
            print(f"running sweep example for N_train = {N_train}")
            sys.stdout.flush()
            stats = self.run_datasweep_sample(hps, N_train)
            save_file = file_utils.sweep_summary_file(config.study_name,
                                                      N_train)
            with open(save_file, 'wb') as f:
                pickle.dump(stats, f)
        print("done! ")
        sys.stdout.flush()

    def run_datasweep_sample(self, hps: Dict, N_train: int) -> None:
        sample_experiment_config = self.load_config_from_suggestion(hps)
        sample_experiment_config.data_config.n_train = N_train
        experiment = self.config.experiment_type(sample_experiment_config)

        def epoch_cb(epoch: int, model: Module, train_loss: float,
                     best_valid_loss: float) -> None:
            pass

        _, best_valid_loss, learned_system = experiment.train(epoch_cb)
        return experiment.evaluation(learned_system)

    def is_complete(self, study: optuna.study.Study) -> bool:
        trials = study.trials
        completed = [trial for trial in trials if
                     trial.state == optuna.trial.TrialState.COMPLETE or trial.state == optuna.trial.TrialState.PRUNED]
        return len(completed) >= self.config.n_trials

    def stop_if_complete(self, study: optuna.study.Study,
                         trial: optuna.trial._frozen.FrozenTrial) -> None:
        if self.is_complete(study):
            study.stop()
        return

    def optimize_hyperparameters(self) -> None:
        config = self.config
        optimizer_config = config.default_experiment_config.optimizer_config

        pruner = optuna.pruners.HyperbandPruner(
            min_resource=config.min_resource,
            max_resource=optimizer_config.epochs
        )
        if config.remote_storage:
            study = optuna.create_study(
                direction="minimize",
                pruner=pruner,
                study_name=config.study_name,
                load_if_exists=True
            )
        else:
            study = optuna.create_study(
                direction="minimize",
                pruner=pruner,
                study_name=config.study_name
            )
        if not self.is_complete(study):
            optuna.logging.get_logger("optuna").addHandler(
                logging.StreamHandler(sys.stdout))
            study.optimize(
                self.optimize,
                n_trials=config.n_trials,
                callbacks=[self.stop_if_complete]
            )
        print("Study completed!")
        self.best_params = study.best_params
        print(study.best_value)
        self.save_hpopt()
        return study.best_params

    def save_hpopt(self) -> None:
        file = file_utils.hyperparameter_file(self.config.study_name)
        with open(file, 'w') as f:
            json.dump(self.best_params, f)

    def get_hpopt(self) -> None:
        file = file_utils.hyperparameter_file(self.config.study_name)
        if os.path.exists(file):
            with open(file, 'r') as f:
                hps = json.load(f)
            return hps
        else:
            return self.optimize_hyperparameters()


if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.DEBUG)

    stiffness = 2500
    study_name = f'mujoco_cube_{stiffness}'
    print(f'running study {study_name}')
    sys.stdout.flush()
    experiment_config = MuJoCoDataExperimentConfig(
        stiffness=float(stiffness),
        study=study_name
    )

    study_config = StudyConfig(
        study_name=study_name,
        default_experiment_config=experiment_config,
        experiment_type=MuJoCoDataExperiment
    )

    # print(asdict(study_config))
    study = Study(study_config)
    study.study()

    # sweep test
    '''
    import os
    stiffness = 300
    POP = 32
    TSKIP = 16
    V200 = False
    CUBE_XML = 'assets/cube_mujoco.xml'
    study_name = f'mujoco_cube_{stiffness}_sweep_test'
    os.system(f'rm -r results/{study_name}')

    optimizer_config = OptimizerConfig(
        lr = 1e-4,
        wd = 0.,
        patience = 0
        )
    
    learnable_config = DeepLearnableSystemConfig(T_history = 1)
    data_config = DataConfig(
        N_train = POP - 4,
        N_valid = 2,
        N_test = 2,
        T_skip = TSKIP#,
        #static_noise = torch.zeros(12),
        #dynamic_noise = torch.zeros(12)
        )

    experiment_config = MuJoCoDataExperimentConfig(
        xml = CUBE_XML,
        learnable_config = learnable_config,
        optimizer_config = optimizer_config,
        data_config = data_config,
        stiffness = stiffness,
        v200 = V200,
        study = study_name,
        N_pop = POP
        )

    study_config = StudyConfig(
        n_trials = 1,
        study_name = study_name,
        remote_storage = False,
        default_experiment_config = experiment_config,
        experiment_type = MuJoCoDataExperiment,
        log_data_size_range = (2,4)
        )

    #print(asdict(study_config))
    study = Study(study_config)
    study.study()
    study.study()
    '''
