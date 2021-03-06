"""Wrappers for Drake/ContactNets multibody experiments."""
from dataclasses import field, dataclass
from enum import Enum
from typing import List, Optional, cast, Dict

import torch
from torch import Tensor

from dair_pll.deep_learnable_system import DeepLearnableExperiment
from dair_pll.drake_system import DrakeSystem
from dair_pll.experiment import SupervisedLearningExperiment, \
    SystemConfig, \
    SupervisedLearningExperimentConfig
from dair_pll.multibody_learnable_system import \
    MultibodyLearnableSystem
from dair_pll.system import System


@dataclass
class DrakeSystemConfig(SystemConfig):
    urdfs: Dict[str, str] = field(default_factory=dict)


class MultibodyLosses(Enum):
    PREDICTION_LOSS = 1
    CONTACTNETS_LOSS = 2


@dataclass
class MultibodyLearnableSystemConfig(DrakeSystemConfig):
    loss: MultibodyLosses = MultibodyLosses.PREDICTION_LOSS
    """Whether to use ContactNets or prediction loss."""


class DrakeExperiment(SupervisedLearningExperiment):
    drake_system: Optional[DrakeSystem]

    def __init__(self, config: SupervisedLearningExperimentConfig) -> None:
        super().__init__(config)
        self.drake_system = None

    def get_drake_system(self) -> DrakeSystem:
        if not hasattr(self, 'drake_system') or self.drake_system is None:
            base_config = cast(DrakeSystemConfig, self.config.base_config)
            dt = self.config.data_config.dt
            self.drake_system = DrakeSystem(base_config.urdfs, dt)
        return self.drake_system

    def get_base_system(self) -> System:
        return self.get_drake_system()


class DrakeDeepLearnableExperiment(DrakeExperiment, DeepLearnableExperiment):
    pass


class DrakeMultibodyLearnableExperiment(DrakeExperiment):

    def __init__(self, config: SupervisedLearningExperimentConfig) -> None:
        super().__init__(config)
        learnable_config = cast(MultibodyLearnableSystemConfig,
                                self.config.learnable_config)
        if learnable_config.loss == MultibodyLosses.CONTACTNETS_LOSS:
            self.loss_callback = self.contactnets_loss

    def get_learned_system(self, _: List[Tensor]) -> MultibodyLearnableSystem:
        learnable_config = cast(MultibodyLearnableSystemConfig,
                                self.config.learnable_config)
        return MultibodyLearnableSystem(learnable_config.urdfs,
                                        self.config.data_config.dt)

    def contactnets_loss(self,
                         x_past: Tensor,
                         x_future: Tensor,
                         system: System,
                         keep_batch: bool = False) -> Tensor:
        r""" :py:data:`~dair_pll.experiment.LossCallbackCallable`
        which applies the ContactNets [1] loss to the system.

        References:
            [1] S. Pfrommer*, M. Halm*, and M. Posa. "ContactNets: Learning
            Discontinuous Contact Dynamics with Smooth, Implicit
            Representations," Conference on Robotic Learning, 2020,
            https://proceedings.mlr.press/v155/pfrommer21a.html
        """
        assert isinstance(system, MultibodyLearnableSystem)
        x = x_past[..., -1, :]
        # pylint: disable=E1103
        u = torch.zeros(x.shape[:-1] + (0,))
        x_plus = x_future[..., 0, :]
        loss = system.contactnets_loss(x, u, x_plus)
        if not keep_batch:
            loss = loss.mean()
        return loss
