r"""Classes for generating and managing datasets for experiments.

Centers around the :class:`ExperimentDataManager` type, which transforms a
set of trajectories saved to disk for various tasks encountered during an
experiment."""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, cast

import torch
from torch import Tensor
from torch.utils.data import Dataset

from dair_pll import file_utils


@dataclass
class TrajectorySliceConfig:
    """:func:`~dataclasses.dataclass` for configuring a trajectory slicing
    for training process."""
    t_skip: int = 0
    """Index of first time to predict from ``>=`` :attr:`t_history` ``- 1``."""
    t_history: int = 1
    r"""Number of steps in initial condition for prediction, ``>= 1``\ ."""
    t_prediction: int = 1
    r"""Number of future steps to use during training/evaluation, ``>= 1``\ ."""

    def __post_init__(self):
        """Method to check validity of parameters."""
        assert self.t_skip + 1 >= self.t_history
        assert self.t_history >= 1
        assert self.t_prediction >= 1


@dataclass
class DataConfig:
    """:func:`~dataclasses.dataclass` for configuring a trajectory dataset."""
    dt: float = 1e-3
    r"""Time step, ``> 0``\ ."""
    train_fraction: float = 0.5
    r"""Fraction of training trajectories to select, ``<= 1, >= 0``\ ."""
    valid_fraction: float = 0.25
    r"""Fraction of validation trajectories to select, ``<= 1, >= 0``\ ."""
    test_fraction: float = 0.25
    r"""Fraction of testing trajectories to select, ``<= 1, >= 0``\ ."""
    slice_config: TrajectorySliceConfig = field(
        default_factory=TrajectorySliceConfig)
    update_dynamically: bool = False
    """Whether to check for new trajectories after each epoch."""

    def __post_init__(self):
        """Method to check validity of parameters."""
        fractions = [
            self.train_fraction, self.valid_fraction, self.test_fraction
        ]
        assert all(0. <= fraction <= 1. for fraction in fractions)
        assert sum(fractions) <= 1


class TrajectorySliceDataset(Dataset):
    r"""Dataset of trajectory slices for training.

    Given a list of trajectories and a :py:class:`TrajectorySliceConfig`\ ,
    generates sets of (previous states, future states) transition pairs to be
    used with the training loss of an experiment.

    Extends :py:class:`torch.utils.data.Dataset` type in order to be managed
    in the training process with a :py:class:`torch.utils.data.DataLoader`\ .
    """
    config: TrajectorySliceConfig
    """Slice configuration describing durations and start index."""
    previous_states_slices: List[Tensor]
    r"""Initial conditions of duration ``self.config.t_history`` ."""
    future_states_slices: List[Tensor]
    r"""Future targets of duration ``self.config.t_prediction`` ."""

    def __init__(self, config: TrajectorySliceConfig):
        """
        Args:
            config: configuration object for slice dataset.
        """
        self.config = config
        self.previous_states_slices = []  # type: List[Tensor]
        self.future_states_slices = []  # type: List[Tensor]

    def add_slices_from_trajectory(self, trajectory: Tensor) -> None:
        """Incorporate trajectory into dataset as a set of slices.

        Args:
            trajectory: ``(T, *)`` state trajectory.
        """
        trajectory_duration = trajectory.shape[0]
        first_time_index = self.config.t_skip
        last_time_index = trajectory_duration - self.config.t_prediction
        previous_states_length = self.config.t_history
        future_states_length = self.config.t_prediction
        assert first_time_index <= last_time_index
        for time in range(first_time_index, last_time_index):
            self.previous_states_slices.append(
                trajectory[(time + 1 - previous_states_length):(time + 1), :])
            self.future_states_slices.append(
                trajectory[(time + 1):(time + 1 + future_states_length), :])

    def __len__(self) -> int:
        """Length of dataset as number of total slice pairs."""
        return len(self.previous_states_slices)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        """Retrieve slice pair at index."""
        return self.previous_states_slices[idx], self.future_states_slices[idx]


@dataclass
class TrajectorySet:
    """Dataclass encapsulating the various transforms of a set of
    trajectories that are used during the training and evaluation process,
    including:

        * Slices for training;
        * Entire trajectories for evaluation; and
        * Indices associated with on-disk location for experiment resumption.
    """
    slices: TrajectorySliceDataset
    """Trajectories rendered as a dataset of time slices."""
    trajectories: List[Tensor] = field(default_factory=lambda: [])
    """Trajectories in their raw format."""
    indices: Tensor = field(default_factory=lambda: Tensor([]).long())
    """Indices associated with on-disk filenames."""

    def __post_init__(self):
        """Validate correspondence between trajectories and indices."""
        assert self.indices.nelement() == len(self.trajectories)
        # assure all indices are unique
        assert self.indices.unique().nelement() == self.indices.nelement()

    def add_trajectories(self, trajectory_list: List[Tensor], indices: Tensor) \
            -> None:
        """Add new subset of trajectories to set.

        Args:
            trajectory_list: List of new ``(T, *)`` state trajectories.
            indices: indices associated with on-disk filenames.
        """
        self.trajectories.extend(trajectory_list)
        for trajectory in trajectory_list:
            self.slices.add_slices_from_trajectory(trajectory)
        # pylint: disable=no-member
        self.indices = torch.cat([self.indices, indices])


class ExperimentDataManager:
    r"""Management object for maintaining training, validation, and testing
    data for an experiment.

    Loads trajectories stored in standard location associated with provided
    storage directory; splits into train/valid/test sets; and instantiates
    transformations for each set of data as a :py:class:`TrajectorySet`\ .
    """
    trajectory_dir: str
    config: DataConfig
    train_set: TrajectorySet
    valid_set: TrajectorySet
    test_set: TrajectorySet
    n_sorted: int

    def __init__(self, storage: str, config: DataConfig,
                 initial_split: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
                 use_ground_truth: bool = False) -> None:
        """s
        Args:
            storage: Storage directory to source trajectories from.
            config: Configuration object.
            initial_split: Initial apportionment of trajectories into (train,
              valid, test) sets.
            use_ground_truth: Whether trajectories should be sourced from
              ground-truth or learning data.
        """
        if use_ground_truth:
            self.trajectory_dir = file_utils.ground_truth_data_dir(storage)
        else:
            self.trajectory_dir = file_utils.learning_data_dir(storage)
        self.config = config
        self.train_set = self.make_empty_trajectory_set()
        self.valid_set = self.make_empty_trajectory_set()
        self.test_set = self.make_empty_trajectory_set()
        self.n_sorted = 0
        if initial_split:
            self.extend_trajectory_sets(initial_split)

    @property
    def _trajectory_sets(
            self) -> Tuple[TrajectorySet, TrajectorySet, TrajectorySet]:
        """getter for tuple of (train, valid, test) set."""
        return self.train_set, self.valid_set, self.test_set

    def trajectory_set_indices(self) -> Tuple[Tensor, Tensor, Tensor]:
        """The sets of indices associated with the (train, valid,
        test) trajectories."""
        index_lists = [
            trajectory_set.indices for trajectory_set in self._trajectory_sets
        ]
        return cast(Tuple[Tensor, Tensor, Tensor], tuple(index_lists))

    def make_empty_trajectory_set(self) -> TrajectorySet:
        r"""Instantiates an empty :py:class:`TrajectorySet` associated with
        the time slice configuration contained in :py:attr:`config`\ ."""
        slice_dataset = TrajectorySliceDataset(self.config.slice_config)
        return TrajectorySet(slices=slice_dataset)

    def extend_trajectory_sets(
            self, index_lists: Tuple[Tensor, Tensor, Tensor]) -> None:
        """Supplement each of (train, valid, test) trajectory sets with
        provided trajectories, listed by their on-disk indices.

        Args:
            index_lists: Lists of trajectory indices for each set.
        """
        for trajectory_set, trajectory_indices in zip(self._trajectory_sets,
                                                      index_lists):
            trajectories = [
                torch.load(
                    file_utils.trajectory_file(self.trajectory_dir,
                                               trajectory_index))
                for trajectory_index in trajectory_indices
            ]
            trajectory_set.add_trajectories(trajectories, trajectory_indices)
            self.n_sorted += trajectory_indices.nelement()

    def get_updated_trajectory_sets(
            self) -> Tuple[TrajectorySet, TrajectorySet, TrajectorySet]:
        """Returns an up-to-date partition of trajectories on disk.

        Checks if some trajectories on disk have yet to be sorted,
        and supplements the (train, valid, test) sets with these additional
        trajectories before returning the updated sets.

        Returns:
            Training set.
            Validation set.
            Test set.
        """
        config = self.config
        n_on_disk = file_utils.get_trajectory_count(self.trajectory_dir)
        if n_on_disk != self.n_sorted:
            n_unsorted = n_on_disk - self.n_sorted
            n_train = round(n_unsorted * config.train_fraction)
            n_valid = round(n_unsorted * config.valid_fraction)
            n_remaining = n_unsorted - n_valid - n_train
            n_test = min(n_remaining, round(n_unsorted * config.test_fraction))

            n_requested = n_train + n_valid + n_test
            assert n_requested <= n_unsorted

            # pylint: disable=no-member
            trajectory_order = torch.randperm(n_unsorted) + self.n_sorted
            train_indices = trajectory_order[:n_train]
            trajectory_order = trajectory_order[n_train:]

            valid_indices = trajectory_order[:n_valid]
            trajectory_order = trajectory_order[n_valid:]
            test_indices = trajectory_order[:n_test]

            self.extend_trajectory_sets(
                (train_indices, valid_indices, test_indices))

        return self._trajectory_sets
