"""Wrappers for Drake/ContactNets multibody experiments."""
import time
from dataclasses import field, dataclass
from enum import Enum
from typing import List, Optional, cast, Dict
import pdb

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from dair_pll.dataset_management import TrajectorySliceDataset, TrajectorySet
from dair_pll.deep_learnable_system import DeepLearnableExperiment
from dair_pll.drake_system import DrakeSystem
from dair_pll.experiment import SupervisedLearningExperiment, \
    SystemConfig, \
    SupervisedLearningExperimentConfig, \
    LOGGING_DURATION
from dair_pll.multibody_learnable_system import \
    MultibodyLearnableSystem, \
    W_COMP, W_PEN, W_DISS
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
    inertia_mode: int = 4
    """What inertial parameters to learn."""


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
                                        self.config.data_config.dt,
                                        learnable_config.inertia_mode)

    def write_to_tensorboard(self, epoch: int, learned_system: System,
                             statistics: Dict) -> None:
        """In addition to extracting and writing training progress summary via
        the parent :py:meth:`Experiment.write_to_tensorboard` method, also make
        a breakdown plot of loss contributions for the ContactNets loss
        formulation.

        Args:
            epoch: Current epoch.
            learned_system: System being trained.
            statistics: Summary statistics for learning process.
        """
        assert self.tensorboard_manager is not None

        # Begin recording wall-clock logging time.
        start_log_time = time.time()
        epoch_vars, system_summary = self.build_epoch_vars_and_system_summary(
                                            learned_system, statistics)

        # Start computing individual loss components.
        # First get a batch sized portion of the shuffled training set.
        train_traj_set, _, _ = self.data_manager.get_trajectory_split()
        train_dataloader = DataLoader(
            train_traj_set.slices,
            batch_size=self.config.optimizer_config.batch_size.value,
            shuffle=True)

        # Calculate the average loss components.
        losses_pred, losses_comp, losses_pen, losses_diss = [], [], [], []
        for xy_i in train_dataloader:
            x_i: Tensor = xy_i[0]
            y_i: Tensor = xy_i[1]

            x = x_i[..., -1, :]
            x_plus = y_i[..., 0, :]
            u = torch.zeros(x.shape[:-1] + (0,))

            loss_pred, loss_comp, loss_pen, loss_diss = \
                learned_system.calculate_contactnets_loss_terms(x, u, x_plus)

            losses_pred.append(loss_pred.clone().detach())
            losses_comp.append(loss_comp.clone().detach())
            losses_pen.append(loss_pen.clone().detach())
            losses_diss.append(loss_diss.clone().detach())

        # Calculate average and scale by hyperparameter weights.
        avg_loss_pred = cast(Tensor, sum(losses_pred) \
                            / len(losses_pred)).mean()
        avg_loss_comp = W_COMP*cast(Tensor, sum(losses_comp) \
                            / len(losses_comp)).mean()
        avg_loss_pen = W_PEN*cast(Tensor, sum(losses_pen) \
                            / len(losses_pen)).mean()
        avg_loss_diss = W_DISS*cast(Tensor, sum(losses_diss) \
                            / len(losses_diss)).mean()

        avg_loss_total = torch.sum(avg_loss_pred + avg_loss_comp + \
                                   avg_loss_pen + avg_loss_diss)

        loss_breakdown = {'loss_total': avg_loss_total,
                          'loss_pred': avg_loss_pred,
                          'loss_comp': avg_loss_comp,
                          'loss_pen': avg_loss_pen,
                          'loss_diss': avg_loss_diss}

        # Include the loss breakdown into system summary.
        system_summary.overlaid_scalars = [loss_breakdown]
        
        # Overwrite the logging time.
        logging_duration = time.time() - start_log_time
        epoch_vars[LOGGING_DURATION] = logging_duration

        self.tensorboard_manager.update(epoch, epoch_vars,
                                        system_summary.videos,
                                        system_summary.meshes,
                                        system_summary.overlaid_scalars)

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
