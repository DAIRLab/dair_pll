from abc import ABC
from dataclasses import dataclass
from typing import Tuple, Type, Optional, cast

import pdb

import torch
from torch import Tensor
from torch.nn import Module

from dair_pll.deep_learnable_model import DeepLearnableModel, DeepRecurrentModel
from dair_pll.experiment import SupervisedLearningExperiment, \
    SupervisedLearningExperimentConfig
from dair_pll.experiment_config import SystemConfig
from dair_pll.integrator import Integrator, VelocityIntegrator, \
    PartialStepCallback
from dair_pll.state_space import StateSpace
from dair_pll.system import System


@dataclass
class DeepLearnableSystemConfig(SystemConfig):
    integrator_type: Type[Integrator] = VelocityIntegrator
    layers: int = 1
    nonlinearity: Module = torch.nn.ReLU
    hidden_size: int = 128
    model_constructor: Type[DeepLearnableModel] = DeepRecurrentModel


class DeepLearnableSystem(System):
    model: Module

    def __init__(self,
                 base_system: System,
                 config: DeepLearnableSystemConfig,
                 training_data: Optional[Tensor] = None) -> None:
        space = base_system.space
        output_size = config.integrator_type.calc_out_size(space)

        model = config.model_constructor(space.n_x, config.hidden_size,
                                         output_size, config.layers,
                                         config.nonlinearity)
        if not (training_data is None):
            model.set_normalization(training_data)

        integrator = config.integrator_type(space, model,
                                            base_system.integrator.dt)

        super().__init__(space, integrator)
        self.model = model
        self.set_carry_sampler(lambda: torch.zeros((1, 1, config.hidden_size)))

    def preprocess_initial_condition(self, x_0: Tensor,
                                     carry_0: Tensor) -> Tuple[Tensor, Tensor]:
        """Preload initial condition."""
        if len(x_0.shape) > 1 and x_0.shape[1] > 1:
            # recurrent start, preload trajectory
            x_pre = x_0[..., :(-1), :]
            _, carry_0 = self.model.sequential_eval(x_pre, carry_0)
            return x_0[..., (-1):, :], carry_0
        else:
            return x_0, carry_0


class DeepLearnableExperiment(SupervisedLearningExperiment, ABC):

    def get_learned_system(self, train_states: Tensor) -> System:
        deep_learnable_config = cast(DeepLearnableSystemConfig,
                                     self.config.learnable_config)

        return DeepLearnableSystem(self.get_base_system(),
                                   deep_learnable_config, train_states)
