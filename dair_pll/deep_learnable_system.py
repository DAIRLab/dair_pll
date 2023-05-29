from abc import ABC
from dataclasses import dataclass, field
from typing import Tuple, Type, Optional, cast

import torch
from torch import Tensor
from torch.nn import Module

from dair_pll.deep_learnable_model import DeepLearnableModel, DeepRecurrentModel
from dair_pll.experiment import SupervisedLearningExperiment
from dair_pll.experiment_config import SystemConfig
from dair_pll.hyperparameter import Int
from dair_pll.integrator import Integrator, VelocityIntegrator
from dair_pll.system import System


@dataclass
class DeepLearnableSystemConfig(SystemConfig):
    integrator_type: Type[Integrator] = field(
        default_factory=lambda: VelocityIntegrator)
    layers: Int = Int(4, log=False)
    nonlinearity: Type[Module] = field(default_factory=lambda: torch.nn.ReLU)
    hidden_size: Int = Int(64, log=True)
    model_constructor: Type[DeepLearnableModel] = field(
        default_factory=lambda: DeepRecurrentModel)


class DeepLearnableSystem(System):
    model: DeepLearnableModel

    def __init__(self,
                 base_system: System,
                 config: DeepLearnableSystemConfig,
                 training_data: Optional[Tensor] = None) -> None:
        space = base_system.space
        output_size = config.integrator_type.calc_out_size(space)
        # pdb.set_trace()
        model = config.model_constructor(space.n_x, config.hidden_size.value,
                                         output_size, config.layers.value,
                                         config.nonlinearity)
        if not (training_data is None):
            # pdb.set_trace()
            model.set_normalization(training_data)

        integrator = config.integrator_type(space, model,
                                            base_system.integrator.dt)

        super().__init__(space, integrator)
        self.model = model
        if issubclass(config.model_constructor, DeepRecurrentModel):
            self.set_carry_sampler(lambda: torch.zeros(
                (1, config.hidden_size.value)))

    def preprocess_initial_condition(self, x_0: Tensor,
                                     carry_0: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Preprocesses initial condition state sequence into single state
        initial condition for integration.

        For example, an RNN would use the state sequence to "preload" hidden
        states in the RNN, where ``carry_0`` would provide an initial hidden
        state, and the output would be the hidden state after the RNN
        receives the state sequence.

        Args:
            x_0: ``(*, T_0, space.n_x)`` initial state sequence.
            carry_0: ``(*, ?)`` initial hidden state.

        Returns:
            ``(*, space.n_x)`` processed initial state.
            ``(*, ?)`` processed initial hidden state.
        """
        assert len(x_0.shape) > 1
        if x_0.shape[1] > 1:
            # recurrent start, preload trajectory
            x_pre = x_0[..., :(-1), :]
            _, carry_0 = self.model.sequential_eval(x_pre, carry_0)
            return x_0[..., -1:, :], carry_0
        else:
            return x_0[..., -1, :], carry_0


class DeepLearnableExperiment(SupervisedLearningExperiment, ABC):

    def get_learned_system(self, train_states: Tensor) -> System:
        deep_learnable_config = cast(DeepLearnableSystemConfig,
                                     self.config.learnable_config)
        return DeepLearnableSystem(self.get_base_system(),
                                   deep_learnable_config, train_states)
