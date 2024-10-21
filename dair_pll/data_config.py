from dataclasses import dataclass, field
from typing import List


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
    his_state_keys: List[str] = field(default_factory=list)
    r"""If set, interpret input as TensorDict and use these keys for history."""
    pred_state_keys: List[str] = field(default_factory=list)
    r"""If set, interpret input as TensorDict and use these keys for prediction."""
    shuffle: bool = True
    r"""Whether to shuffle data during training."""

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
    slice_config: TrajectorySliceConfig = field(default_factory=TrajectorySliceConfig)
    r"""Config for arranging trajectories into times slices for training."""
    update_dynamically: bool = False
    """Whether to check for new trajectories after each epoch."""

    def __post_init__(self):
        """Method to check validity of parameters."""
        fractions = [self.train_fraction, self.valid_fraction, self.test_fraction]
        assert all(0.0 <= fraction <= 1.0 for fraction in fractions)
        assert sum(fractions) <= 1
