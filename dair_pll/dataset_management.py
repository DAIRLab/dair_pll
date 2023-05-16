r"""Classes for generating and managing datasets for experiments.

Centers around the :class:`ExperimentDataManager` type, which transforms a
set of trajectories saved to disk for various tasks encountered during an
experiment."""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, cast, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset

from dair_pll import file_utils

TrainValidTestFractions = Tuple[float, float, float]
TrainValidTestCounts = Tuple[int, int, int]
TrainValidTestQuantities = Union[TrainValidTestFractions, TrainValidTestCounts]
ThreeTensorTuple = Tuple[Tensor, Tensor, Tensor]


def _assert_valid_train_valid_test_quantities(
        quantities: TrainValidTestQuantities) -> None:
    assert len(quantities) == 3
    if isinstance(quantities[0], float):
        assert all(isinstance(quantity, float) for quantity in quantities)
        assert all(0. <= quantity <= 1. for quantity in quantities)
        assert sum(quantities) <= 1
    if isinstance(quantities[0], int):
        assert all(isinstance(quantity, int) for quantity in quantities)
        assert all(quantity >= 0 for quantity in quantities)


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
    train_valid_test_quantities: TrainValidTestQuantities = (0.5, 0.25, 0.25)
    r"""Fractions of on-disk total or raw size of train/valid/test sets."""
    slice_config: TrajectorySliceConfig = field(
        default_factory=TrajectorySliceConfig)
    r"""Config for arranging trajectories into times slices for training."""
    update_dynamically: bool = False
    """Whether to check for new trajectories after each epoch."""
    exclude_mask: Optional[Tensor] = None
    """List of trajectories to exclude from dataset."""
    use_ground_truth: bool = False
    """Whether to use ground truth data for training/evaluation."""

    def __post_init__(self):
        """Method to check validity of parameters."""
        _assert_valid_train_valid_test_quantities(
            self.train_valid_test_quantities)
        assert self.dt > 0.

    def split_ratios(self):
        """Relative portions of train/valid/test sets."""
        if isinstance(self.train_valid_test_quantities[0], float):
            return self.train_valid_test_quantities

        return tuple(quantity / sum(self.train_valid_test_quantities)
                     for quantity in self.train_valid_test_quantities)


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
            self.previous_states_slices.append(
                trajectory[(index + 1 - previous_states_length):(index + 1), :])
            self.future_states_slices.append(
                trajectory[(index + 1):(index + 1 + future_states_length), :])

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
    """Number of files split into (train, valid, test) or excluded so far."""

    def __init__(self,
                 storage: str,
                 config: DataConfig,
                 initial_split: Optional[ThreeTensorTuple] = None) -> None:
        """
        Args:
            storage: Storage directory to source trajectories from.
            config: Configuration object.
            initial_split: Optionally, lists of trajectory indices that
              should be sorted into (train, valid, test) sets from the
              beginning.
        """
        if config.use_ground_truth:
            self.trajectory_dir = file_utils.ground_truth_data_dir(storage)
        else:
            self.trajectory_dir = file_utils.learning_data_dir(storage)
        self.config = config
        self.train_set = self.make_empty_trajectory_set()
        self.valid_set = self.make_empty_trajectory_set()
        self.test_set = self.make_empty_trajectory_set()
        self.n_sorted = 0
        if initial_split:
            # first, check that initial split does not contain excluded indices.
            n_on_disk = file_utils.get_trajectory_count(self.trajectory_dir)
            for split in initial_split:
                assert split.max() < n_on_disk
                if config.exclude_mask is not None:
                    assert not torch.any(config.exclude_mask[split])
            self.extend_trajectory_sets(initial_split)

    def _get_indices_to_split(self) -> Tensor:
        """Indices of trajectories available and needed for T/V/T splot on
        disk."""
        n_on_disk = file_utils.get_trajectory_count(self.trajectory_dir)
        all_indices_available = torch.arange(self.n_sorted, n_on_disk).long()

        # Remove excluded indices.
        exclude_mask = self.config.exclude_mask
        if exclude_mask is not None:
            # extend mask
            extension = torch.zeros(n_on_disk - exclude_mask.nelement(),
                                    dtype=torch.bool)
            exclude_mask = torch.cat((exclude_mask, extension))
            keep_mask = ~exclude_mask[all_indices_available]
            all_indices_available = all_indices_available[keep_mask]

        return all_indices_available

    def _get_incremental_split(self) -> ThreeTensorTuple:
        """New indices to split into T/V/T sets."""
        all_indices_to_split = self._get_indices_to_split()

        # permute addable indices
        n_to_split = all_indices_to_split.nelement()
        all_indices_to_split = all_indices_to_split[torch.randperm(n_to_split)]

        split_ratios = self.config.split_ratios()
        is_fractions = isinstance(self.config.train_valid_test_quantities[0],
                                  float)
        if not is_fractions:
            totals_and_current_indices = zip(
                self.config.train_valid_test_quantities,
                self.trajectory_set_indices())

            remaining_needed = torch.tensor([
                total_needed - current_indices.nelement()
                for total_needed, current_indices in totals_and_current_indices
            ],
                                            dtype=torch.long)

        if is_fractions or remaining_needed.sum() > n_to_split:
            # if we do not enough trajectories on disk to satisfy the totals,
            # split by ratios
            n_train = int(n_to_split * split_ratios[0])
            n_valid = int(n_to_split * split_ratios[1])
            n_test = n_to_split - n_train - n_valid
            # TODO: properly handle case where sum(ratios) < 1.
            # Currently an issue as n_sorted is not updated properly to
            # reflect skipped trajectories.
            n_added_to_splits = torch.tensor([n_train, n_valid, n_test],
                                    dtype=torch.long)
        else:
            # else, split by total remaining needed.
            n_added_to_splits = remaining_needed

        breaks = torch.cat((torch.zeros(1, dtype=torch.long),
                            torch.cumsum(n_added_to_splits, dim=0).long()))

        split = tuple(all_indices_to_split[int(breaks[i]):int(breaks[i + 1])]
                      for i in range(3))

        return cast(ThreeTensorTuple, split)

    @property
    def _trajectory_sets(
            self) -> Tuple[TrajectorySet, TrajectorySet, TrajectorySet]:
        """Getter for tuple of (train, valid, test) set."""
        return self.train_set, self.valid_set, self.test_set

    def trajectory_set_indices(self) -> ThreeTensorTuple:
        """The sets of indices associated with the (train, valid,
        test) trajectories."""
        index_lists = [
            trajectory_set.indices for trajectory_set in self._trajectory_sets
        ]
        return cast(ThreeTensorTuple, tuple(index_lists))

    def make_empty_trajectory_set(self) -> TrajectorySet:
        r"""Instantiates an empty :py:class:`TrajectorySet` associated with
        the time slice configuration contained in :py:attr:`config`\ ."""
        slice_dataset = TrajectorySliceDataset(self.config.slice_config)
        return TrajectorySet(slices=slice_dataset)

    def extend_trajectory_sets(
            self, index_lists: ThreeTensorTuple) -> None:
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

        Checks if some trajectories on disk should be added to the split,
        and supplements the (train, valid, test) sets with these additional
        trajectories before returning the updated sets.

        Returns:
            Training set.
            Validation set.
            Test set.
        """
        indices_to_split = self._get_incremental_split()
        if any(indices.nelement() > 0 for indices in indices_to_split):
            self.extend_trajectory_sets(indices_to_split)

        return self._trajectory_sets
