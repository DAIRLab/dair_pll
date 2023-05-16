"""Configuration dataclasses for experiments."""
from dataclasses import dataclass, field
from typing import Type, Optional

import torch
from torch.optim import Optimizer

from dair_pll.dataset_management import DataConfig
from dair_pll.hyperparameter import Float, Int
from dair_pll.summary_statistics_constants import LOSS_NAME


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
    validation_variable: str = LOSS_NAME
    """Which quantity to use for early stopping (default: validation loss)."""
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
    #  pylint: disable=too-many-instance-attributes
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
    storage: str = './'
    """Folder for results/data storage. Defaults to working directory."""
    run_name: str = 'experiment_run'
    """Unique identifier for experiment run."""
    run_wandb: bool = True
    """Whether to run Weights and Biases logging."""
    wandb_project: Optional[str] = None
    r"""If :py:attr:`run_wandb`\ , a project to store results under on W&B."""
    full_evaluation_period: int = 1
    """How many epochs should pass between full evaluations."""
    full_evaluation_samples: int = 5
    """How many trajectories to save in full for experiment's summary."""
    update_geometry_in_videos: bool = False
    """Whether to use learned geometry in rollout videos, primarily for
    debugging purposes."""

    def __post_init__(self):
        """Method to check validity of parameters."""
        if self.run_wandb:
            assert self.wandb_project is not None

        if self.full_evaluation_period > 1:
            raise NotImplementedError(
                "Patience not correctly implemented for sporadic evaluation.")
