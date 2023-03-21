"""Defines interfaces for various learning experiments to be run.

Current supported experiment types include:

    * :py:class:`SupervisedLearningExperiment`: An experiment where a
      :py:class:`~dair_pll.system.System` is learned to mimic a
      dataset of trajectories.

"""
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional, Dict, cast, Type, Union
import pdb

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dair_pll import file_utils
from dair_pll.dataset_management import SystemDataManager, \
    DataConfig, TrajectorySliceDataset, TrajectorySet
from dair_pll.hyperparameter import Float, Int
from dair_pll.state_space import StateSpace
from dair_pll.system import System, SystemSummary
from dair_pll.tensorboard_manager import TensorboardManager

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

AVERAGE_TAG = 'mean'

EVALUATION_VARIABLES = [LOSS_NAME, TRAJECTORY_ERROR_NAME]


@dataclass
class SystemConfig:
    """Dummy base :py:class:`~dataclasses.dataclass` for parameters for
    learning dynamics; all inheriting classes are expected to contain all
    necessary configuration attributes."""


@dataclass
class OptimizerConfig:
    """:func:`~dataclasses.dataclass` defining setup and usage opf a Pytorch
    :func:`~torch.optim.Optimizer` for learning."""
    optimizer: Type[Optimizer] = torch.optim.Adam
    """Subclass of :py:class:`~torch.optim.Optimizer` to use."""
    lr: Float = Float(1e-5, log=True)
    """Learning rate."""
    wd: Float = Float(4e-5, log=True)
    """Weight decay."""
    epochs: int = 10000
    """Maximum number of epochs to optimize."""
    patience: int = 30
    """Number of epochs to wait for early stopping."""
    batch_size: Int = Int(64, log=True)
    """Size of batch for an individual gradient step."""


@dataclass
class SupervisedLearningExperimentConfig:
    """:py:class:`~dataclasses.dataclass` defining setup of a
    :py:class:`SupervisedLearningExperiment`"""
    data_config: DataConfig = field(default_factory=DataConfig)
    """Configuration for experiment's
    :py:class:`~dair_pll.system_data_manager.SystemDataManager`."""
    base_config: SystemConfig = field(default_factory=SystemConfig)
    """Configuration for experiment's "base" system, from which trajectories
    are modeled and optionally generated."""
    learnable_config: SystemConfig = field(default_factory=SystemConfig)
    """Configuration for system to be learned."""
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    """Configuration for experiment's optimization process."""
    run_tensorboard: bool = True
    """Whether to run Tensorboard logging."""
    full_evaluation_period: int = 1
    """How many epochs should pass between full evaluations."""
    full_evaluation_samples: int = 5
    """How many trajectories to save in full for experiment's summary."""
    gen_videos: bool = False
    """Whether to use ``VideoWriter`` to generate toss videos."""
    update_geometry_in_videos: bool = False
    """Whether to use learned geometry in rollout videos, primarily for
    debugging purposes.  Does nothing if ``gen_videos == True``."""


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
LossCallbackCallable = Callable[[Tensor, Tensor, System, Optional[bool]],
                                Tensor]
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
    :class:`~dair_pll.system_data_manager.SystemDataManager`
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
    data_manager: SystemDataManager
    """Data manager constructed from details in :attr:`config`"""
    space: StateSpace
    """State space of experiment, inferred from base system."""
    loss_callback: Optional[LossCallbackCallable]
    """Callback function for loss, defaults to prediction loss."""
    tensorboard_manager: Optional[TensorboardManager]
    """Optional tensorboard interface."""

    def __init__(self, config: SupervisedLearningExperimentConfig) -> \
            None:
        super().__init__()

        self.config = config
        file_utils.assure_storage_tree_created(config.data_config.storage)
        base_system = self.get_base_system()
        self.space = base_system.space
        self.data_manager = SystemDataManager(base_system, config.data_config)
        self.loss_callback = cast(LossCallbackCallable, self.prediction_loss)
        # if config.run_tensorboard:
        self.tensorboard_manager = TensorboardManager(
            self.data_manager.get_tensorboard_folder(),
            log_only = not config.run_tensorboard)

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
                                        data_config.t_prediction)
        future = prediction[..., 1:, :]
        return future

    def trajectory_predict(self, x: List[Tensor],
                           system: System) -> Tuple[List[Tensor], List[Tensor]]:
        """Predict from full lists of trajectories.

        Preloads initial conditions from the first ``t_skip + 1`` elements of
        each trajectory.

        Args:
            x: List of ``(T, space.n_x)`` trajectories.
            system: System to run prediction on.

        Returns:
            List of ``(T - t_skip - 1, space.n_x)`` predicted trajectories.

            List of ``(T - t_skip - 1, space.n_x)`` target trajectories.

        """
        t_skip = self.config.data_config.t_skip
        t_begin = t_skip + 1
        x_0 = [x_i[..., :t_begin, :] for x_i in x]
        targets = [x_i[..., t_begin:, :] for x_i in x]
        prediction_horizon = [x_i.shape[-2] - t_skip - 1 for x_i in x]

        assert system.carry_callback is not None
        carry_0 = system.carry_callback()
        predictions = [
            system.simulate(x_0_i, carry_0, horizon_i)[0][..., 1:, :]
            for x_0_i, horizon_i in zip(x_0, prediction_horizon)
        ]
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

    def train_epoch(self, data: DataLoader, system: System,
                    optimizer: Optimizer) -> Tensor:
        """Train learned model for a single epoch.

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
            optimizer.zero_grad()
            loss = self.batch_loss(x_i, y_i, system)
            losses.append(loss.clone().detach())
            loss.backward()
            optimizer.step()
        avg_loss = cast(Tensor, sum(losses) / len(losses))
        return avg_loss

    def calculate_loss_no_grad_step(self, data: DataLoader, system: System) \
        -> Tensor:
        """Evaluate learned model, without taking any gradient steps.

        Args:
            data: Training dataset.
            system: System to be trained.
            optimizer: Optimizer which trains system.

        Returns:
            Scalar average training loss observed during evaluation.
        """
        losses = []
        for xy_i in data:
            x_i: Tensor = xy_i[0]
            y_i: Tensor = xy_i[1]
            loss = self.batch_loss(x_i, y_i, system)
            losses.append(loss.clone().detach())
        avg_loss = cast(Tensor, sum(losses) / len(losses))
        return avg_loss


    def build_epoch_vars_and_system_summary(self, learned_system: System,
                                            statistics: Dict) -> \
                                            Tuple[Dict, SystemSummary]:
        """Extracts and writes summary of training progress to Tensorboard.

        Args:
            learned_system: System being trained.
            statistics: Summary statistics for learning process.

        Returns:
            Scalars dictionary.
            Videos and meshes packaged into a ``SystemSummary``.
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

        system_summary = learned_system.summary(statistics,
            videos=self.config.gen_videos,
            new_geometry=self.config.update_geometry_in_videos)

        epoch_vars.update(system_summary.scalars)
        logging_duration = time.time() - start_log_time
        statistics[LOGGING_DURATION] = logging_duration
        epoch_vars.update(
            {duration: statistics[duration] for duration in ALL_DURATIONS})

        return epoch_vars, system_summary

    def write_to_tensorboard(self, epoch: int, learned_system: System,
                             statistics: Dict) -> None:
        """Extracts and writes summary of training progress to Tensorboard.

        Args:
            epoch: Current epoch.
            learned_system: System being trained.
            statistics: Summary statistics for learning process.
        """
        assert self.tensorboard_manager is not None

        epoch_vars, system_summary = self.build_epoch_vars_and_system_summary(
                                            learned_system, statistics)
        
        self.tensorboard_manager.update(epoch, epoch_vars,
                                        system_summary.videos,
                                        system_summary.meshes)

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
        start_eval_time = time.time()
        statistics = {}
        if (epoch % self.config.full_evaluation_period) == 0:
            train_set, valid_set, _ = self.data_manager.get_trajectory_split()
            n_train_eval = min(len(train_set.trajectories),
                              self.config.full_evaluation_samples)

            n_valid_eval = min(len(valid_set.trajectories),
                               self.config.full_evaluation_samples)

            # Already calculated the training loss, so just grab a "dummy"
            # portion of the training set (the first trajectory) to speed up the
            # computation.
            dummy_train_slice_set = TrajectorySliceDataset(
                train_set.trajectories[:1])

            train_eval_set = TrajectorySet(
                trajectories=train_set.trajectories[:n_train_eval],
                slices=dummy_train_slice_set)
            valid_eval_set = TrajectorySet(
                trajectories=valid_set.trajectories[:n_valid_eval],
                slices=valid_set.slices)

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

        if self.tensorboard_manager is not None:
            self.write_to_tensorboard(epoch, learned_system, statistics)

        # pylint: disable=E1103
        # valid_loss_key = f'{VALID_SET}_{LEARNED_SYSTEM_NAME}_{LOSS_NAME}' \
        #                  f'_{AVERAGE_TAG}'
        # Use validation set mean rollout error as validation loss.
        valid_loss_key = f'{VALID_SET}_{LEARNED_SYSTEM_NAME}' \
                         + f'_{TRAJECTORY_ERROR_NAME}_{AVERAGE_TAG}'
        valid_loss = 0.0 \
            if not valid_loss_key in statistics \
            else statistics[valid_loss_key]
        return torch.tensor(valid_loss)

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

            Fully-trained system, with parameters corresponding to
        """

        # get train/test/val trajectories
        train_set, _, _ = \
            self.data_manager.get_trajectory_split()

        # Prepare sets for training.
        train_dataloader = DataLoader(
            train_set.slices,
            batch_size=self.config.optimizer_config.batch_size.value,
            shuffle=True)

        # Setup optimization.
        # pylint: disable=E1103
        learned_system = self.get_learned_system(
            torch.cat(train_set.trajectories))
        optimizer = self.get_optimizer(learned_system)

        if self.tensorboard_manager is not None:
            self.tensorboard_manager.launch()

        # Track epochs since best validation-set loss has been seen, and save
        # model parameters from that epoch.
        # pylint: disable=E1103
        epochs_since_best = 0
        best_learned_system_state = deepcopy(learned_system.state_dict())

        training_loss = self.calculate_loss_no_grad_step(train_dataloader,
                                                              learned_system)
        best_valid_loss = self.per_epoch_evaluation(0, learned_system,
                                                    training_loss, 0.)
        epoch_callback(0, learned_system, training_loss, best_valid_loss)

        for epoch in range(1, self.config.optimizer_config.epochs + 1):
            if self.config.data_config.dynamic_updates_from is not None:
                # reload training data

                # get train/test/val trajectories
                train_set, _, _ = \
                    self.data_manager.get_trajectory_split()

                # Prepare sets for training.
                train_dataloader = DataLoader(
                    train_set.slices,
                    batch_size=self.config.optimizer_config.batch_size.value,
                    shuffle=True)

            learned_system.train()
            start_train_time = time.time()
            training_loss = self.train_epoch(train_dataloader, learned_system,
                                             optimizer)
            training_duration = time.time() - start_train_time
            learned_system.eval()
            valid_loss = self.per_epoch_evaluation(epoch, learned_system,
                                                   training_loss,
                                                   training_duration)

            # Check for validation loss improvement.
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_learned_system_state = deepcopy(
                    learned_system.state_dict())
                epochs_since_best = 0
            else:
                epochs_since_best += 1

            # Decide to early-stop or not.
            if epochs_since_best >= self.config.optimizer_config.patience:
                break

            epoch_callback(epoch, learned_system, training_loss,
                           best_valid_loss)

        # Reload best parameters.
        learned_system.load_state_dict(best_learned_system_state)

        # kill tensorboard.
        print("killing tboard")
        if self.tensorboard_manager is not None:
            self.tensorboard_manager.stop()

        return training_loss, best_valid_loss, learned_system

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
                                       shuffle=True)
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
                    trajectories, system)
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
                aux_comps = space.auxiliary_comparisons()
                for comp_name in aux_comps:
                    stats[f'{set_name}_{system_name}_{comp_name}'] = to_json([
                        aux_comps[comp_name](tp, tt)
                        for tp, tt in zip(traj_pred, traj_target)
                    ])

        summary_stats = {}  # type: StatisticsDict
        for key, stat in stats.items():
            if isinstance(stat, np.ndarray):
                if len(stats) > 0:
                    if isinstance(stat[0], float):
                        summary_stats[f'{key}_{AVERAGE_TAG}'] = np.average(stat)

        stats.update(summary_stats)
        return stats

    def evaluation(self, learned_system: System) -> StatisticsDict:
        r"""Evaluate both oracle and learned system on training, validation,
        and testing data.

        Implemented as a wrapper for :meth:`evaluate_systems_on_sets`.

        Args:
            learned_system: Trained system.

        Returns:
            Statistics dictionary.

        Warnings:
            Currently assumes prediction horizon of 1.
        """
        sets = dict(zip(ALL_SETS, self.data_manager.get_trajectory_split()))
        systems = {
            ORACLE_SYSTEM_NAME: self.get_oracle_system(),
            LEARNED_SYSTEM_NAME: learned_system
        }
        return self.evaluate_systems_on_sets(systems, sets)
        