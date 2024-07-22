"""Defines interfaces for various learning experiments to be run.

Current supported experiment types include:

    * :py:class:`SupervisedLearningExperiment`: An experiment where a
      :py:class:`~dair_pll.system.System` is learned to mimic a
      dataset of trajectories.

"""
import dataclasses
import signal
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
import pdb
from typing import Any, List, Tuple, Callable, Optional, Dict, cast, Union

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dair_pll import file_utils
from dair_pll.dataset_management import ExperimentDataManager, \
    TrajectorySet
from dair_pll.experiment_config import SupervisedLearningExperimentConfig
from dair_pll.multibody_learnable_system import MultibodyLearnableSystem
from dair_pll.state_space import StateSpace, FloatingBaseSpace
from dair_pll.system import System, SystemSummary
from dair_pll.wandb_manager import WeightsAndBiasesManager

# Enable default_collate for TensorDict
from tensordict.tensordict import TensorDict
def collate_tensordict_fn(batch, *, collate_fn_map: Optional[Any] = None):
    out = None
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum(x.numel() for x in batch)
        storage = elem._typed_storage()._new_shared(numel, device=elem.device)
        out = elem.new(storage).resize_(len(batch), *list(elem.size()))
    return torch.stack(batch, 0, out=out)
torch.utils.data._utils.collate.default_collate_fn_map[TensorDict] = collate_tensordict_fn


@dataclass
class TrainingState:
    """Dataclass to store a complete summary of the state of training
    process."""
    # pylint: disable=too-many-instance-attributes
    trajectory_set_split_indices: Tuple[Tensor, Tensor, Tensor]
    """Which trajectory indices are in train/valid/test sets."""
    best_learned_system_state: dict
    """State of learned system when it had the best validation loss so far."""
    current_learned_system_state: dict
    """Current state of learned system."""
    optimizer_state: dict
    r"""Current state of training :py:class:`torch.optim.Optimizer`\ ."""
    epoch: int = 1
    """Current epoch."""
    epochs_since_best: int = 0
    """Number of epochs since best validation loss so far was achieved."""
    best_valid_loss: Tensor = field(default_factory=lambda: torch.tensor(1e10))
    """Value of best validation loss so far."""
    wandb_run_id: Optional[str] = None
    """If using W&B, the ID of the run associated with this experiment."""
    finished_training: bool = False
    """Whether training has finished."""


TRAIN_SET = 'train'
VALID_SET = 'valid'
TEST_SET = 'test'

TRAIN_TIME_SETS = [TRAIN_SET, VALID_SET]
ALL_SETS = [TRAIN_SET, VALID_SET, TEST_SET]

TRAINING_DURATION = 'training_duration'
EVALUATION_DURATION = 'evaluation_duration'
LOGGING_DURATION = 'logging_duration'
ALL_DURATIONS = [TRAINING_DURATION, EVALUATION_DURATION, LOGGING_DURATION]

MAX_SAVED_TRAJECTORIES = 5

BASE_SYSTEM_NAME = 'base'
ORACLE_SYSTEM_NAME = 'oracle'
LEARNED_SYSTEM_NAME = 'model'

LOSS_NAME = 'loss'
TRAJECTORY_ERROR_NAME = 'trajectory_mse'
PREDICTED_VELOCITY_SIZE = 'v_plus_squared'
DELTA_VELOCITY_SIZE = 'delta_v_squared'
TARGET_NAME = 'target_sample'
PREDICTION_NAME = 'prediction_sample'
TRAJECTORY_POSITION_ERROR_NAME = 'pos_int_traj'
TRAJECTORY_ROTATION_ERROR_NAME = 'angle_int_traj'
TRAJECTORY_PENETRATION_NAME = 'penetration_int_traj'
RESIDUAL_SINGLE_STEP_SIZE_NAME = 'residual_norm_stepwise'
RESIDUAL_TRAJECTORY_SIZE_MSE_NAME = 'residual_norm_traj_mse'

AVERAGE_TAG = 'mean'

EVALUATION_VARIABLES = [LOSS_NAME, TRAJECTORY_ERROR_NAME, 
    TRAJECTORY_POSITION_ERROR_NAME, TRAJECTORY_ROTATION_ERROR_NAME,
    TRAJECTORY_PENETRATION_NAME, RESIDUAL_SINGLE_STEP_SIZE_NAME,
    RESIDUAL_TRAJECTORY_SIZE_MSE_NAME
]


#:
EpochCallbackCallable = Callable[[int, System, Tensor, Tensor], None]
"""Type hint for extra callback to be called on each epoch of training.

Args:
    epoch: Current epoch.
    learned_system: Partially-trained learned system.
    train_loss: Current epoch's average training loss.
    best_valid_loss: Best validation loss so far.
"""

#:
LossCallbackCallable = Callable[[Tensor, Tensor, System, bool], Tensor]
"""Callback to evaluate loss on batch of trajectory slices.

By default, set to prediction loss (
:meth:`SupervisedLearningExperiment.prediction_loss`)

Args:
    x_past: ``(*,t_history,space.n_x)`` previous states in slice.
    x_future:  ``(*,t_prediction,space.n_x)`` future states in slice.
    system: system on which to evaluate loss
    keep_batch: whether or not to collapse batch into a single scalar.
Returns:
    ``(*,)`` or scalar loss.
"""


def default_epoch_callback(epoch: int, _learned_system: System,
                           train_loss: Tensor, best_valid_loss: Tensor) -> None:
    """Default :py:data:`EpochCallbackCallable` which prints epoch, training
    loss, and best validation loss so far."""
    print(epoch, train_loss, best_valid_loss)


StatisticsValue = Union[List, float, np.ndarray]
StatisticsDict = Dict[str, StatisticsValue]


class SupervisedLearningExperiment(ABC):
    r"""Supervised learning experiment.

    Implements the training and evaluation processes for a supervised
    learning experiment, where a :class:`~dair_pll.system.System` is
    learned to capture a dataset of trajectories.

    The dataset of trajectories is encapsulated in a
    :class:`~dair_pll.dataset_management.ExperimentDataManager`
    object. This dataset is either stored to disc by the user,
    or alternatively is generated from the experiment's *base system*\ .

    The *base system*\ is a :class:`~dair_pll.system.System` with
    the same :class:`~dair_pll.state_space.StateSpace` as the
    system to be learned.

    Training is completed via a Pytorch :class:`~torch.optim.Optimizer`.

    The training process keeps track of various statistics about the learning
    process, and optionally logs the learned system's
    :class:`~dair_pll.system.SystemSummary` to Tensorboard on each
    epoch.
    """
    config: SupervisedLearningExperimentConfig
    """Configuration of the experiment."""
    space: StateSpace
    """State space of experiment, inferred from base system."""
    loss_callback: Optional[LossCallbackCallable]
    """Callback function for loss, defaults to prediction loss."""
    wandb_manager: Optional[WeightsAndBiasesManager]
    """Optional tensorboard interface."""
    learning_data_manager: Optional[ExperimentDataManager]
    """Manager of trajectory data used in learning process."""

    def __init__(self, config: SupervisedLearningExperimentConfig) -> None:
        super().__init__()

        self.config = config
        file_utils.assure_storage_tree_created(config.storage)
        if not hasattr(self, 'space'):
            base_system = self.get_base_system()
            self.space = base_system.space
        self.loss_callback = cast(LossCallbackCallable, self.prediction_loss)
        self.learning_data_manager = None

        file_utils.save_configuration(config.storage, config.run_name, config)

    @abstractmethod
    def get_base_system(self) -> System:
        """Abstract callback function to construct base system from system
        config.

        Returns:
            Experiment's base system.
        """

    def get_oracle_system(self) -> System:
        """Abstract callback function to construct oracle system for
        experiment.

        Conceptually, the oracle system is an ideal system to compare the
        learned system against. By default, the oracle system is simply the
        base system. However, in some scenarios, a different type of oracle
        is appropriate. For example, if the learned system is recurrent,
        the oracle system might most appropriately take a recurrent slice of
        initial states, process them with a Kalman Filter for the base
        system, and then predict the future.

        Returns:
            Experiment's oracle system.
        """
        return self.get_base_system()

    @abstractmethod
    def get_learned_system(self, train_states: Tensor) -> System:
        """Abstract callback function to construct learnable system for
        experiment.

        Optionally, learned system can be initialized to depend on the
        training dataset.

        Args:
            train_states: ``(*, space.n_x)`` batch of all states in training
              set.
        Returns:
            Experiment's learnable system.
        """

    def get_optimizer(self, learned_system: System) -> Optimizer:
        """Constructs optimizer for experiment.

        Args:
            learned_system: System to be trained.

        Returns:
            Optimizer for training.
        """
        config = self.config.optimizer_config
        if issubclass(config.optimizer, torch.optim.Adam):
            return config.optimizer(learned_system.parameters(),
                                    lr=config.lr.value,
                                    weight_decay=config.wd.value)
        raise TypeError('Unsupported optimizer type:',
                        config.optimizer.__name__)

    def batch_predict(self, x_past: Tensor, system: System) -> Tensor:
        """Predict forward in time from initial conditions.

        Args:
            x_past: ``(*, t_history, space.n_x)`` batch of initial states.
            system: System to run prediction on.

        Returns:
            ``(*, t_prediction, space.n_x)`` batch of predicted future states.
        """
        data_config = self.config.data_config

        # pylint: disable=E1103
        assert system.carry_callback is not None
        carries = torch.stack([system.carry_callback() for _ in x_past])
        prediction, _ = system.simulate(x_past, carries,
                                        data_config.slice_config.t_prediction)
        future = prediction[..., 1:, :]
        return future

    def trajectory_predict(
            self,
            x: List[Tensor],
            system: System,
            do_detach: bool = False) -> Tuple[List[Tensor], List[Tensor]]:
        """Predict from full lists of trajectories.

        Preloads initial conditions from the first ``t_skip + 1`` elements of
        each trajectory.

        Args:
            x: List of ``(*, T, space.n_x)`` trajectories.
            system: System to run prediction on.
            do_detach: Whether to detach each prediction from the computation
              graph; useful for memory management for large groups of
              trajectories.

        Returns:
            List of ``(*, T - t_skip - 1, space.n_x)`` predicted trajectories.

            List of ``(*, T - t_skip - 1, space.n_x)`` target trajectories.

        """
        t_skip = self.config.data_config.slice_config.t_skip
        t_begin = t_skip + 1
        x_0 = [x_i[..., :t_begin, :] for x_i in x]
        targets = [x_i[..., t_begin:, :] for x_i in x]
        prediction_horizon = [x_i.shape[-2] - t_skip - 1 for x_i in x]

        assert system.carry_callback is not None
        carry_0 = system.carry_callback()
        predictions = []
        for x_0_i, horizon_i, target_i in zip(x_0, prediction_horizon, targets):
            target_shape = target_i.shape

            x_prediction_i, carry_i = system.simulate(x_0_i, carry_0, horizon_i)
            del carry_i
            to_append = x_prediction_i[..., 1:, :].reshape(target_shape)
            if do_detach:
                predictions.append(to_append.detach().clone())
                del x_prediction_i
            else:
                predictions.append(to_append)
        return predictions, targets

    def prediction_loss(self,
                        x_past: Tensor,
                        x_future: Tensor,
                        system: System,
                        keep_batch: bool = False) -> Tensor:
        r"""Default :py:data:`LossCallbackCallable` which evaluates to system's
        :math:`l_2` prediction error on batch:

        .. math::

            \mathcal{L}(x_{p,i,\cdot}, x_{f,i,\cdot}) = \sum_{j} ||\hat x_{f,
            i,j} - x_{f,i,j}||^2,

        where :math:`x_{p,i,\cdot}, x_{f,i,\cdot}` are the :math:`i`\ th
        elements of the past and future batches; and
        :math:`\hat x_{f,i,j}` is the :math:`j`-step forward prediction of the
        model from the past batch.

        See :py:data:`LossCallbackCallable` for additional type signature info.
        """
        space = self.space
        x_predicted = self.batch_predict(x_past, system)
        v_future = space.v(x_future)
        v_predicted = space.v(x_predicted)
        avg_const = v_predicted.nelement() // v_predicted.shape[0]
        if not keep_batch:
            avg_const *= x_predicted.shape[0]
        return space.velocity_square_error(v_future, v_predicted,
                                           keep_batch) / avg_const

    def batch_loss(self,
                   x_past: Tensor,
                   x_future: Tensor,
                   system: System,
                   keep_batch: bool = False) -> Tensor:
        """Runs :py:attr:`loss_callback` (a
        :py:data:`LossCallbackCallable`) on the given batch."""
        assert self.loss_callback is not None
        return self.loss_callback(x_past, x_future, system, keep_batch)

    def train_epoch(self,
                    data: DataLoader,
                    system: System,
                    optimizer: Optional[Optimizer] = None) -> Tensor:
        """Train learned model for a single epoch.  Takes gradient steps in the
        learned parameters if ``optimizer`` is provided.

        Args:
            data: Training dataset.
            system: System to be trained.
            optimizer: Optimizer which trains system.

        Returns:
            Scalar average training loss observed during epoch.
        """
        losses = []
        for xy_i in data:
            x_i: Tensor = xy_i[0]
            y_i: Tensor = xy_i[1]

            if optimizer is not None:
                optimizer.zero_grad()
            loss = self.batch_loss(x_i, y_i, system)
            losses.append(loss.clone().detach())

            if optimizer is not None:
                loss.backward()
                optimizer.step()

        avg_loss = cast(Tensor, sum(losses) / len(losses))
        return avg_loss

    def base_and_learned_comparison_summary(
            self, statistics: Dict, learned_system: System) -> SystemSummary:
        """Extracts a :py:class:`~dair_pll.system.SystemSummary` that compares
        the base system to the learned system.

        Args:
            statistics: Dictionary of training statistics.
            learned_system: Most updated version of system during training.

        Returns:
            Summary of comparison between systems.
        """
        # pylint: disable=unused-argument
        return SystemSummary()

    def build_epoch_vars_and_system_summary(self, statistics: Dict,
        learned_system: System, skip_videos=True) -> Tuple[Dict, SystemSummary]:
        """Build epoch variables and system summary for learning process.

        Args:
            statistics: Summary statistics for learning process.
            learned_system: System being trained.
            skip_videos: Whether to skip making videos or not.

        Returns:
            Dictionary of scalars to log.
            System summary.
        """
        # begin recording wall-clock logging time.
        start_log_time = time.time()

        epoch_vars = {}
        for stats_set in TRAIN_TIME_SETS:
            for variable in EVALUATION_VARIABLES:
                var_key = f'{stats_set}_{LEARNED_SYSTEM_NAME}' + \
                          f'_{variable}_{AVERAGE_TAG}'
                if var_key in statistics:
                    epoch_vars[f'{stats_set}_{variable}'] = statistics[var_key]

        learned_system_summary = learned_system.summary(statistics)

        if not skip_videos:
            comparison_summary = self.base_and_learned_comparison_summary(
                statistics, learned_system)

        epoch_vars.update(learned_system_summary.scalars)
        logging_duration = time.time() - start_log_time
        statistics[LOGGING_DURATION] = logging_duration
        epoch_vars.update(
            {duration: statistics[duration] for duration in ALL_DURATIONS})

        if not skip_videos:
            epoch_vars.update(comparison_summary.scalars)
            learned_system_summary.videos.update(comparison_summary.videos)
            learned_system_summary.meshes.update(comparison_summary.meshes)

        return epoch_vars, learned_system_summary

    def write_to_wandb(self, epoch: int, learned_system: System,
                       statistics: Dict) -> None:
        """Extracts and writes summary of training progress to Tensorboard.

        Args:
            epoch: Current epoch.
            learned_system: System being trained.
            statistics: Summary statistics for learning process.
        """
        assert self.wandb_manager is not None

        # To save space on W&B storage, only generate comparison videos at first
        # and best epoch, the latter of which is implemented in
        # :meth:`_evaluation`.
        skip_videos = False  #if epoch==0 else True BIBIT temporary for debugging

        epoch_vars, learned_system_summary = \
            self.build_epoch_vars_and_system_summary(statistics, learned_system,
                                                     skip_videos=skip_videos)

        self.wandb_manager.update(epoch, epoch_vars,
                                  learned_system_summary.videos,
                                  learned_system_summary.meshes)

    def per_epoch_evaluation(self, epoch: int, learned_system: System,
                             train_loss: Tensor,
                             training_duration: float) -> Tensor:
        """Evaluates and logs training progress at end of an epoch.

        Runs evaluation on full slice datasets, as well as a handful of
        trajectories.

        Optionally logs the results on tensorboard via
        :meth:`write_to_tensorboard`.

        Args:
            epoch: Current epoch.
            learned_system: System being trained.
            train_loss: Scalar training loss of epoch.
            training_duration: Duration of epoch training in seconds.

        Returns:
            Scalar validation set loss.
        """
        # pylint: disable=too-many-locals
        assert self.learning_data_manager is not None
        start_eval_time = time.time()
        statistics = {}

        if (epoch % self.config.full_evaluation_period) == 0:
            train_set, valid_set, _ = \
                self.learning_data_manager.get_updated_trajectory_sets()

            n_train_eval = min(len(train_set.trajectories),
                               self.config.full_evaluation_samples)

            n_valid_eval = min(len(valid_set.trajectories),
                               self.config.full_evaluation_samples)

            train_eval_set = \
                self.learning_data_manager.make_empty_trajectory_set()
            train_eval_set.add_trajectories(
                train_set.trajectories[:n_train_eval],
                train_set.indices[:n_train_eval])

            valid_eval_set = \
                self.learning_data_manager.make_empty_trajectory_set()
            valid_eval_set.add_trajectories(
                valid_set.trajectories[:n_valid_eval],
                valid_set.indices[:n_valid_eval])

            statistics = self.evaluate_systems_on_sets(
                {LEARNED_SYSTEM_NAME: learned_system}, {
                    TRAIN_SET: train_eval_set,
                    VALID_SET: valid_eval_set
                })

        statistics[f'{TRAIN_SET}_{LEARNED_SYSTEM_NAME}_'
                   f'{LOSS_NAME}_{AVERAGE_TAG}'] = float(train_loss.item())

        statistics[TRAINING_DURATION] = training_duration
        statistics[EVALUATION_DURATION] = time.time() - start_eval_time

        self.statistics = statistics

        if self.wandb_manager is not None:
            self.write_to_wandb(epoch, learned_system, statistics)

        # pylint: disable=E1103
        valid_loss_key = f'{VALID_SET}_{LEARNED_SYSTEM_NAME}_{LOSS_NAME}' \
                         f'_{AVERAGE_TAG}'
        # # Use validation set mean rollout error as validation loss.
        # valid_loss_key = f'{VALID_SET}_{LEARNED_SYSTEM_NAME}' \
        #                  + f'_{TRAJECTORY_ERROR_NAME}_{AVERAGE_TAG}'
        valid_loss = 0.0 \
            if valid_loss_key not in statistics \
            else statistics[valid_loss_key]
        return torch.tensor(valid_loss)

    def setup_training(self) -> Tuple[System, Optimizer, TrainingState]:
        r"""Sets up initial condition for training process.

        Attempts to load initial condition from disk as a
        :py:class:`TrainingState`\ . Otherwise, a fresh training process is
        started.

        Returns:
            Initial learned system.
            Pytorch optimizer.
            Current state of training process.
        """
        is_resumed = False
        training_state = None
        checkpoint_filename = file_utils.get_model_filename(
            self.config.storage, self.config.run_name)
        try:
            # if a checkpoint is saved from disk, attempt to load it.
            checkpoint_dict = torch.load(checkpoint_filename)
            training_state = TrainingState(**checkpoint_dict)
            print("Resumed from disk.")
            is_resumed = True
            self.learning_data_manager = ExperimentDataManager(
                self.config.storage, self.config.data_config,
                training_state.trajectory_set_split_indices)
        except FileNotFoundError:
            self.learning_data_manager = ExperimentDataManager(
                self.config.storage, self.config.data_config)

        train_set, _, _ = \
            self.learning_data_manager.get_updated_trajectory_sets()

        # Setup optimization.
        # pylint: disable=E1103
        learned_system = self.get_learned_system(
            torch.cat(train_set.trajectories))
        optimizer = self.get_optimizer(learned_system)

        if is_resumed:
            assert training_state is not None
            learned_system.load_state_dict(
                training_state.current_learned_system_state)
            optimizer.load_state_dict(training_state.optimizer_state)
        else:
            training_state = TrainingState(
                self.learning_data_manager.trajectory_set_indices(),
                deepcopy(learned_system.state_dict()),
                deepcopy(learned_system.state_dict()),
                deepcopy(optimizer.state_dict()))

            # Our Weights & Biases logic assumes that if there's no training
            # state on disk, that resumption is not allowed. Therefore, we
            # never want to launch wandb_manager without a training state
            # saved to disk.
            torch.save(dataclasses.asdict(training_state), checkpoint_filename)

        if self.config.run_wandb:
            assert self.config.wandb_project is not None
            wandb_directory = file_utils.wandb_dir(self.config.storage,
                                                   self.config.run_name)

            self.wandb_manager = WeightsAndBiasesManager(
                self.config.run_name, wandb_directory,
                self.config.wandb_project, training_state.wandb_run_id)
            training_state.wandb_run_id = self.wandb_manager.launch()
            self.wandb_manager.log_config(self.config)

        return learned_system, optimizer, training_state

    def train(
        self,
        epoch_callback: EpochCallbackCallable = default_epoch_callback,
    ) -> Tuple[Tensor, Tensor, System]:
        """Run training process for experiment.

        Terminates training with early stopping, parameters for which are set in
        :attr:`config`.

        Args:
            epoch_callback: Callback function at end of each epoch.

        Returns:
            Final-epoch training loss.
            Best-seen validation set loss.
            Fully-trained system, with parameters corresponding to best-seen
              validation loss.
        """
        checkpoint_filename = file_utils.get_model_filename(
            self.config.storage, self.config.run_name)

        learned_system, optimizer, training_state = self.setup_training()
        assert self.learning_data_manager is not None

        train_set, _, _ = \
            self.learning_data_manager.get_updated_trajectory_sets()

        # Prepare sets for training.
        train_dataloader = DataLoader(
            train_set.slices,
            batch_size=self.config.optimizer_config.batch_size.value,
            shuffle=self.config.data_config.slice_config.shuffle,
            generator=torch.Generator(device=torch.get_default_device()),
        )

        # Calculate the training loss before any parameter updates.  Calls
        # ``train_epoch`` without providing an optimizer, so no gradient steps
        # will be taken.
        learned_system.eval()
        training_loss = self.train_epoch(train_dataloader, learned_system)

        print(f"Training Loss: {training_loss}")

        # Terminate if the training state indicates training already finished.
        if training_state.finished_training:
            learned_system.load_state_dict(
                training_state.best_learned_system_state)
            return training_loss, training_state.best_valid_loss, learned_system

        # Report losses before any parameter updates.
        if training_state.epoch == 1:
            print("Report pre-train losses")
            training_state.best_valid_loss = self.per_epoch_evaluation(
                0, learned_system, training_loss, 0.)
            epoch_callback(0, learned_system, training_loss,
                           training_state.best_valid_loss)

        patience = self.config.optimizer_config.patience

        # Start training loop.
        try:
            print("Starting Full Training")
            while training_state.epoch <= self.config.optimizer_config.epochs:
                print(f"Epoch: {training_state.epoch}")
                if self.config.data_config.update_dynamically:
                    # reload training data

                    # get train/test/val trajectories
                    train_set, _, _ = \
                        self.learning_data_manager.get_updated_trajectory_sets()

                    # Prepare sets for training.
                    train_dataloader = DataLoader(
                        train_set.slices,
                        batch_size=self.config.optimizer_config.batch_size.
                        value,
                        shuffle=self.config.data_config.slice_config.shuffle,
                        generator=torch.Generator(device=torch.get_default_device()))

                    training_state.trajectory_set_split_indices = \
                        self.learning_data_manager.trajectory_set_indices()

                learned_system.train()
                start_train_time = time.time()
                training_loss = self.train_epoch(train_dataloader,
                                                 learned_system, optimizer)
                
                training_duration = time.time() - start_train_time
                learned_system.eval()
                valid_loss = self.per_epoch_evaluation(training_state.epoch,
                                                       learned_system,
                                                       training_loss,
                                                       training_duration)

                # Check for validation loss improvement.
                if valid_loss < training_state.best_valid_loss:
                    training_state.best_valid_loss = valid_loss
                    training_state.best_learned_system_state = deepcopy(
                        learned_system.state_dict())
                    training_state.epochs_since_best = 0
                else:
                    training_state.epochs_since_best += 1

                epoch_callback(training_state.epoch, learned_system,
                               training_loss, training_state.best_valid_loss)

                # Decide to early-stop or not.
                if training_state.epochs_since_best >= patience:
                    break

                training_state.current_learned_system_state = \
                    learned_system.state_dict()
                training_state.optimizer_state = optimizer.state_dict()
                training_state.epoch += 1

            # Mark training as completed, whether by early stopping or by
            # reaching the epoch limit.
            training_state.finished_training = True

        finally:
            # this code should execute, even if a program exit is triggered
            # in the above try block.

            # Stop SIGINT (Ctrl+C) from exiting during saving.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            print("Saving training state before exit...")
            torch.save(dataclasses.asdict(training_state), checkpoint_filename)
            signal.signal(signal.SIGINT, signal.SIG_DFL)

        # Reload best parameters.
        #print("Loading best parameters...")
        #learned_system.load_state_dict(training_state.best_learned_system_state)
        #print("Done loading best parameters.")
        return training_loss, training_state.best_valid_loss, learned_system

    def extra_metrics(self) -> Dict[str, Callable[[Tensor, Tensor], Tensor]]:
        return {}

    def evaluate_systems_on_sets(
            self, systems: Dict[str, System],
            sets: Dict[str, TrajectorySet]) -> StatisticsDict:
        r"""Evaluate given systems on trajectory sets.

        Builds a "statistics" dictionary containing a thorough evaluation
        each system on each set, containing the following:

            * Single step and trajectory prediction losses.
            * Squared norms of velocity and delta-velocity (for normalization).
            * Sample target and prediction trajectories.
            * Auxiliary trajectory comparisons defined in 
              :meth:`dair_pll.state_space.StateSpace\
              .auxiliary_comparisons()`
            * Summary statistics of the above where applicable.

        Args:
            systems: Named dictionary of systems to evaluate.
            sets: Named dictionary of sets to evaluate.

        Returns:
            Statistics dictionary.

        Warnings:
            Currently assumes prediction horizon of 1.
        """
        # pylint: disable=too-many-locals

        stats = {}  # type: StatisticsDict
        space = self.space

        def to_json(possible_tensor: Union[float, List, Tensor]) -> \
                StatisticsValue:
            """Converts tensor to :class:`~np.ndarray`, which enables saving
            stats as json."""
            if isinstance(possible_tensor, list):
                return [to_json(value) for value in possible_tensor]
            if torch.is_tensor(possible_tensor):
                tensor = cast(Tensor, possible_tensor)
                return tensor.detach().cpu().numpy()

            assert isinstance(possible_tensor, float)
            return possible_tensor

        for set_name, trajectory_set in sets.items():
            trajectories = trajectory_set.trajectories
            n_saved_trajectories = min(MAX_SAVED_TRAJECTORIES,
                                       len(trajectories))
            slices_loader = DataLoader(trajectory_set.slices,
                                       batch_size=128,
                                       shuffle=False,
                                       generator=torch.Generator(device=torch.get_default_device()))
            slices = trajectory_set.slices[:]
            all_x = cast(List[Tensor], slices[0])
            all_y = cast(List[Tensor], slices[1])

            # hack: assume 1-step prediction for now
            # pylint: disable=E1103
            v_plus = [space.v(y[:1, :]) for y in all_y]
            v_minus = [space.v(x[-1:, :]) for x in all_x]
            dv2 = torch.stack([
                space.velocity_square_error(vp, vm)
                for vp, vm in zip(v_plus, v_minus)
            ])
            vp2 = torch.stack(
                [space.velocity_square_error(vp, 0 * vp) for vp in v_plus])
            stats[f'{set_name}_{DELTA_VELOCITY_SIZE}'] = to_json(dv2)
            stats[f'{set_name}_{PREDICTED_VELOCITY_SIZE}'] = to_json(vp2)

            for system_name, system in systems.items():
                model_loss_list = []
                for batch_x, batch_y in slices_loader:
                    model_loss_list.append(
                        self.prediction_loss(batch_x, batch_y, system, True))
                model_loss = torch.cat(model_loss_list)
                loss_name = f'{set_name}_{system_name}_{LOSS_NAME}'
                stats[loss_name] = to_json(model_loss)

                if system_name == LEARNED_SYSTEM_NAME:
                    trajectories = [t.unsqueeze(0) for t in trajectories]
                traj_pred, traj_target = self.trajectory_predict(
                    trajectories, system, True)
                if system_name == LEARNED_SYSTEM_NAME:
                    traj_target = [t.squeeze(0) for t in traj_target]
                    traj_pred = [t.squeeze(0) for t in traj_pred]
                    stats[f'{set_name}_{system_name}_{TARGET_NAME}'] = \
                        to_json(traj_target[:n_saved_trajectories])
                    stats[f'{set_name}_{system_name}_{PREDICTION_NAME}'] = \
                        to_json(traj_pred[:n_saved_trajectories])

                # pylint: disable=E1103
                trajectory_mse = torch.stack([
                    space.state_square_error(tp, tt)
                    for tp, tt in zip(traj_pred, traj_target)
                ])
                stats[f'{set_name}_{system_name}_{TRAJECTORY_ERROR_NAME}'] = \
                    to_json(trajectory_mse)

                # Add position and rotation error over trajectory.  TODO this
                # could be implemented more elegantly; perhaps somewhere else
                # like in space.auxiliary_comparisons or a child experiment
                # class like DrakeMultibodyLearnableExperiment.
                running_pos_mse = None
                running_angle_mse = None
                for space_i in space.spaces:
                    if isinstance(space_i, FloatingBaseSpace):
                        pos_mse = torch.stack([
                            space_i.base_error(tp, tt)
                            for tp, tt in zip(traj_pred, traj_target)
                        ])
                        angle_mse = torch.stack([
                            space_i.quaternion_error(tp, tt)
                            for tp, tt in zip(traj_pred, traj_target)
                        ])
                        if running_pos_mse == None:
                            running_pos_mse = pos_mse
                            running_angle_mse = angle_mse
                        else:
                            running_pos_mse += pos_mse
                            running_angle_mse += angle_mse

                stats[f'{set_name}_{system_name}_' + \
                      f'{TRAJECTORY_POSITION_ERROR_NAME}'] = \
                    to_json(running_pos_mse)
                stats[f'{set_name}_{system_name}_' + \
                      f'{TRAJECTORY_ROTATION_ERROR_NAME}'] = \
                    to_json(running_angle_mse)

                # Add residual sizes over trajectory and single steps.
                if isinstance(system, MultibodyLearnableSystem):
                    if system.residual_net != None:
                        residual_mse = torch.stack([
                            torch.linalg.norm(system.residual_net(tp),
                                              dim=1).sum()
                            for tp in traj_pred
                        ])
                        stats[f'{set_name}_{system_name}_' + \
                              f'{RESIDUAL_TRAJECTORY_SIZE_MSE_NAME}'] = \
                            to_json(residual_mse/len(traj_pred))

                        residual_single_step_mse = torch.stack([
                            torch.linalg.norm(system.residual_net(x_i),
                                              dim=1).sum()
                            for x_i in all_x
                        ])
                        stats[f'{set_name}_{system_name}_' + \
                              f'{RESIDUAL_SINGLE_STEP_SIZE_NAME}'] = \
                            to_json(residual_mse/len(all_x))

                extra_metrics = self.extra_metrics()
                for metric_name in extra_metrics:
                    stats[f'{set_name}_{system_name}_{metric_name}'] = to_json(
                        torch.tensor([
                        extra_metrics[metric_name](tp, tt)
                        for tp, tt in zip(traj_pred, traj_target)
                    ]))

                aux_comps = space.auxiliary_comparisons()
                for comp_name in aux_comps:
                    stats[f'{set_name}_{system_name}_{comp_name}'] = to_json([
                        aux_comps[comp_name](tp, tt)
                        for tp, tt in zip(traj_pred, traj_target)
                    ])

        summary_stats = {}  # type: StatisticsDict
        for key, stat in stats.items():
            if isinstance(stat, np.ndarray):
                if len(stat) > 0:
                    if isinstance(stat[0], float):
                        summary_stats[f'{key}_{AVERAGE_TAG}'] = np.average(stat)

        stats.update(summary_stats)
        return stats

    def _evaluation(self, learned_system: System) -> StatisticsDict:
        r"""Evaluate both oracle and learned system on training, validation,
        and testing data, and saves results to disk.

        Implemented as a wrapper for :meth:`evaluate_systems_on_sets`.

        Args:
            learned_system: Trained system.

        Returns:
            Statistics dictionary.

        Warnings:
            Currently assumes prediction horizon of 1.
        """
        assert self.learning_data_manager is not None
        sets = dict(
            zip(ALL_SETS,
                self.learning_data_manager.get_updated_trajectory_sets()))
        systems = {
            ORACLE_SYSTEM_NAME: self.get_oracle_system(),
            LEARNED_SYSTEM_NAME: learned_system
        }

        evaluation = self.evaluate_systems_on_sets(systems, sets)
        file_utils.save_evaluation(self.config.storage, self.config.run_name,
                                   evaluation)

        # Generate final toss/geometry inspection videos with best parameters.
        comparison_summary = self.base_and_learned_comparison_summary(
            evaluation, learned_system)
        self.wandb_manager.update(int(1e4), {}, comparison_summary.videos, {})

        return evaluation

    def generate_results(
        self,
        epoch_callback: EpochCallbackCallable = default_epoch_callback,
    ) -> Tuple[System, StatisticsDict]:
        r"""Get the final learned model and results/statistics of experiment.
        Along with the model corresponding to best validation loss, this will
        return previously saved results on disk if they already exist, or run
        the experiment to generate them if they don't.

        Args:
            epoch_callback: Callback function at end of each epoch.

        Returns:
            Fully-trained system, with parameters corresponding to best-seen
              validation loss.
            Statistics dictionary.
        """
        _, _, learned_system = self.train(epoch_callback)

        print("Done Training")

        try:
            print("Looking for previously generated statistics...")
            statistics = file_utils.load_evaluation(self.config.storage,
                                                    self.config.run_name)
            print("Done loading statistics.")
        except FileNotFoundError:
            print("Did not find statistics; generating them... (this could " + \
                  "take several minutes)")
            statistics = self._evaluation(learned_system)
            print("Done generating statistics.")

        return learned_system, statistics
