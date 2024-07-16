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
from dair_pll.data_config import TrajectorySliceConfig, DataConfig


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
        trajectory_length = trajectory.shape[0]
        first_time_index = self.config.t_skip
        last_time_index = trajectory_length - self.config.t_prediction
        previous_states_length = self.config.t_history
        future_states_length = self.config.t_prediction
        assert first_time_index <= last_time_index
        for index in range(first_time_index, last_time_index):
            his_state = trajectory[(index + 1 - previous_states_length):(index + 1)]
            if len(self.config.his_state_keys) > 0:
                # Only keep requested state keys
                for key in [key for key in his_state.keys()]:
                    if key not in self.config.his_state_keys:
                        del his_state[key]
                # TODO: HACK Add Time Index
                his_state["time"] = index * torch.ones([1, 1], dtype=torch.int32)
            self.previous_states_slices.append(his_state)

            pred_state = trajectory[(index + 1):(index + 1 + future_states_length)]
            if len(self.config.pred_state_keys) > 0:
                # Only keep requested state keys
                for key in [key for key in pred_state.keys()]:
                    if key not in self.config.pred_state_keys:
                        del pred_state[key]
                # TODO: HACK Add Time Index
                pred_state["time"] = (index+1) * torch.ones([1, 1], dtype=torch.int32)
            self.future_states_slices.append(pred_state)

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
    indices: Tensor = field(default_factory=lambda: torch.tensor([]).long())
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
    """Directory in which trajectory files are stored."""
    config: DataConfig
    """Configuration for manipulating data."""
    train_set: TrajectorySet
    """Training trajectory set."""
    valid_set: TrajectorySet
    """Validation trajectory set."""
    test_set: TrajectorySet
    """Test trajectory set."""
    n_sorted: int
    """Number of files on disk split into (train, valid, test) sets so far."""

    def __init__(self, storage: str, config: DataConfig,
                 initial_split: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
                 use_ground_truth: bool = False) -> None:
        """
        Args:
            storage: Storage directory to source trajectories from.
            config: Configuration object.
            initial_split: Optionally, lists of trajectory indices that
              should be sorted into (train, valid, test) sets from the
              beginning.
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
