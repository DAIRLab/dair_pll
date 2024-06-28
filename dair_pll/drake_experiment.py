"""Wrappers for Drake/ContactNets multibody experiments."""
import time
from abc import ABC
from dataclasses import field, dataclass
from enum import Enum
from typing import Any, List, Optional, cast, Dict, Callable
import pdb

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from dair_pll import file_utils
from dair_pll import vis_utils
from dair_pll.deep_learnable_system import DeepLearnableExperiment
from dair_pll.drake_system import DrakeSystem
from dair_pll.experiment import SupervisedLearningExperiment, \
    LEARNED_SYSTEM_NAME, PREDICTION_NAME, TARGET_NAME, \
    TRAJECTORY_PENETRATION_NAME, LOGGING_DURATION
from dair_pll.experiment_config import SystemConfig, \
    SupervisedLearningExperimentConfig
from dair_pll.hyperparameter import Float
from dair_pll.multibody_learnable_system import \
    MultibodyLearnableSystem, LOSS_INERTIA_AGNOSTIC, LOSS_BALANCED, \
    LOSS_POWER, LOSS_PLL_ORIGINAL, LOSS_CONTACT_VELOCITY, LOSS_VARIATIONS, \
    LOSS_VARIATION_NUMBERS
from dair_pll.system import System, SystemSummary

@dataclass
class DrakeSystemConfig(SystemConfig):
    urdfs: Dict[str, str] = field(default_factory=dict)
    additional_system_builders: List[str] = field(default_factory=list)
    additional_system_kwargs: List[Dict[str, Any]] = field(default_factory=list)


class MultibodyLosses(Enum):
    PREDICTION_LOSS = 1
    CONTACTNETS_LOSS = 2


@dataclass
class DrakeMultibodyLearnableExperimentConfig(SupervisedLearningExperimentConfig
                                             ):
    visualize_learned_geometry: bool = True
    """Whether to use learned geometry in trajectory overlay visualization."""


@dataclass
class MultibodyLearnableSystemConfig(DrakeSystemConfig):
    loss: MultibodyLosses = MultibodyLosses.PREDICTION_LOSS
    """Whether to use ContactNets or prediction loss."""
    inertia_mode: int = 4
    """What inertial parameters to learn."""
    loss_variation: str = LOSS_POWER
    """What loss variation to use."""
    w_pred: float = 1.0
    """Weight of prediction term in ContactNets loss (suggested keep at 1.0)."""
    w_comp: Float = Float(1e0, log=True)  #1e-1
    """Weight of complementarity term in ContactNets loss."""
    w_diss: Float = Float(1e0, log=True)
    """Weight of dissipation term in ContactNets loss."""
    w_pen: Float = Float(1e0, log=True)  #1e1
    """Weight of penetration term in ContactNets loss."""
    w_res: Float = Float(1e0, log=True)
    """Weight of residual norm in loss."""
    w_res_w: Float = Float(1e0, log=True)
    """Weight of residual weights in loss."""
    do_residual: bool = False
    """Whether to include a residual physics block."""
    network_width: int = 128
    """Width of residual network."""
    network_depth: int = 2
    """Depth of residual network."""
    represent_geometry_as: str = 'box'
    """How to represent geometry (box, mesh, or polygon)."""
    randomize_initialization: bool = True
    """Whether to randomize initialization."""
    g_frac: float = 1.0
    """What fraction of the true gravitational constant to use."""

from functools import partial
from pydrake.all import DiagramBuilder, MultibodyPlant
import importlib
def system_builder_from_string(string: str, **kwargs) -> Callable[[DiagramBuilder, MultibodyPlant], None]:
    """
    Get a function object from a class string
    """
    module_name = string[:string.rfind('.')]
    func_name = string[string.rfind('.')+1:]
    func = getattr(importlib.import_module(module_name), func_name)
    return partial(func, **kwargs)

class DrakeExperiment(SupervisedLearningExperiment, ABC):
    base_drake_system: Optional[DrakeSystem]
    augmented_drake_system: Optional[DrakeSystem]
    visualization_system: Optional[DrakeSystem]

    def __init__(self, config: SupervisedLearningExperimentConfig) -> None:
        super().__init__(config)
        self.base_drake_system = None
        self.visualization_system = None

    def get_drake_system(self) -> DrakeSystem:
        has_property = hasattr(self, 'base_drake_system')
        if not has_property or self.base_drake_system is None:
            base_config = cast(DrakeSystemConfig, self.config.base_config)
            dt = self.config.data_config.dt
            assert len(base_config.additional_system_builders) == len(base_config.additional_system_kwargs), f"Expected {len(base_config.additional_system_builders)} == {len(base_config.additional_system_kwargs)}"
            self.base_drake_system = DrakeSystem(base_config.urdfs, 
                dt, 
                additional_system_builders=[system_builder_from_string(string, **kwargs) for string, kwargs in zip(base_config.additional_system_builders, base_config.additional_system_kwargs)]
            )
        return self.base_drake_system

    def get_base_system(self) -> System:
        return self.get_drake_system()

    def get_augmented_system(self, additional_forces: str) -> DrakeSystem:
        """Get a ``DrakeSystem`` where the Drake multibody plant has additional
        forces in the ``applied_generalized_force`` or ``applied_spatial_force``
        input ports."""
        print("Getting augmented system!")
        has_property = hasattr(self, 'augmented_drake_system')
        if not has_property or self.augmented_drake_system is None:
            base_config = cast(DrakeSystemConfig, self.config.base_config)
            dt = self.config.data_config.dt
            self.augmented_drake_system = DrakeSystem(
                base_config.urdfs, dt, additional_forces=additional_forces)
        return self.augmented_drake_system

    def get_learned_drake_system(
            self, learned_system: System) -> Optional[DrakeSystem]:
        r"""If possible, constructs a :py:class:`DrakeSystem` -equivalent
        model of the given learned system, such as when the learned system is a
        :py:class:`MultibodyLearnableSystem`\ .

        Args:
            learned_system: System being learned in experiment.

        Returns:
            Drake version of learned system.
        """
        return None

    def visualizer_regeneration_is_required(self) -> bool:
        """Checks if visualizer should be regenerated, e.g. if learned
        geometries have been updated and need to be pushed to the visulizer.
        """
        return False

    def get_visualization_system(self, learned_system: System) -> DrakeSystem:
        """Generate a dummy :py:class:`DrakeSystem` for visualizing comparisons
        between trajectories generated by the base system and something else,
        e.g. data.

        Implemented as a thin wrapper of
        ``vis_utils.generate_visualization_system()``, which generates a
        drake system where each model in the base
        :py:class:`DrakeSystem` has a duplicate, and visualization
        elements are repainted for visual distinction.

        Args:
            learned_system: Current trained learnable system.

        Returns:
            New :py:class:`DrakeSystem` with doubled state and repainted
            elements.
        """
        # Generate a new visualization system if it needs to use the updated
        # geometry, or if it hasn't been created yet.
        regeneration_is_required = self.visualizer_regeneration_is_required()
        if regeneration_is_required or self.visualization_system is None:
            visualization_file = file_utils.get_trajectory_video_filename(
                self.config.storage, self.config.run_name)
            base_system = self.get_drake_system()
            self.visualization_system = \
                vis_utils.generate_visualization_system(
                    base_system,
                    visualization_file,
                    learned_system=self.get_learned_drake_system(learned_system)
                )

        return self.visualization_system

    def base_and_learned_comparison_summary(
            self, statistics: Dict, learned_system: System) -> SystemSummary:
        r"""Extracts a :py:class:`~dair_pll.system.SystemSummary` that compares
        the base system to the learned system.

        For Drake-based experiments, this comparison is implemented as
        overlaid videos of corresponding ground-truth and predicted
        trajectories. The nature of this video is described further in
        :py:mod:`dair_pll.vis_utils`\ .

        Additionally, manually defined trajectories are used to show the learned
        geometries.  This is particularly useful for more expressive geometry
        types like meshes.

        Args:
            statistics: Dictionary of training statistics.
            learned_system: Most updated version of learned system during
              training.

        Returns:
            Summary containing overlaid video(s).
        """
        visualization_system = self.get_visualization_system(learned_system)

        space = self.get_drake_system().space
        videos = {}

        # First do overlay prediction videos.
        for traj_num in [0]:
            for set_name in ['train', 'valid']:
                target_key = f'{set_name}_{LEARNED_SYSTEM_NAME}' + \
                             f'_{TARGET_NAME}'
                prediction_key = f'{set_name}_{LEARNED_SYSTEM_NAME}' + \
                                 f'_{PREDICTION_NAME}'
                if not target_key in statistics:
                    continue
                target_trajectory = Tensor(statistics[target_key][traj_num])
                prediction_trajectory = Tensor(
                    statistics[prediction_key][traj_num])
                visualization_trajectory = torch.cat(
                    (space.q(target_trajectory), space.q(prediction_trajectory),
                     space.v(target_trajectory),
                     space.v(prediction_trajectory)), -1)
                video, framerate = vis_utils.visualize_trajectory(
                    visualization_system, visualization_trajectory)
                videos[f'{set_name}_trajectory_prediction_{traj_num}'] = \
                    (video, framerate)

        # Second do geometry inspection videos -- only relevant for model-based.
        if not type(self) == DrakeDeepLearnableExperiment:
            geometry_inspection_traj = \
                vis_utils.get_geometry_inspection_trajectory(learned_system)
            target_trajectory = geometry_inspection_traj
            prediction_trajectory = geometry_inspection_traj
            visualization_trajectory = torch.cat(
                (space.q(target_trajectory), space.q(prediction_trajectory),
                 space.v(target_trajectory), space.v(prediction_trajectory)), -1)
            video, framerate = vis_utils.visualize_trajectory(
                visualization_system, visualization_trajectory)
            videos['geometry_inspection'] = (video, framerate)

        return SystemSummary(scalars={}, videos=videos, meshes={})

    def get_true_geometry_multibody_learnable_system(self
        ) -> MultibodyLearnableSystem:

        has_property = hasattr(self, 'true_geom_multibody_system')
        if not has_property or self.true_geom_multibody_system is None:
            oracle_system = self.get_oracle_system()
            dt = oracle_system.dt
            urdfs = oracle_system.urdfs

            self.true_geom_multibody_system = MultibodyLearnableSystem(
                init_urdfs=urdfs, dt=dt, inertia_mode=0, loss_variation=0,
                w_pred=1.0, w_comp=1.0, w_diss=1.0, w_pen=1.0, w_res=1.0,
                w_res_w=1.0, do_residual=False,
                represent_geometry_as = \
                    self.config.learnable_config.represent_geometry_as,
                randomize_initialization = False)
            
        return self.true_geom_multibody_system

    def penetration_metric(self, x_pred: Tensor, _x_target: Tensor) -> Tensor:
        true_geom_system = self.get_true_geometry_multibody_learnable_system()

        if x_pred.dim() == 1:
            x_pred = x_pred.unsqueeze(0)
        assert x_pred.dim() == 2
        assert x_pred.shape[1] == true_geom_system.space.n_x

        n_steps = x_pred.shape[0]

        phi, _ = true_geom_system.multibody_terms.contact_terms(x_pred)
        phi = phi.detach().clone()
        smallest_phis = phi.min(dim=1).values
        return -smallest_phis[smallest_phis < 0].sum() / n_steps

    def extra_metrics(self) -> Dict[str, Callable[[Tensor, Tensor], Tensor]]:
        # Calculate penetration metric
        return {TRAJECTORY_PENETRATION_NAME: self.penetration_metric}


class DrakeDeepLearnableExperiment(DrakeExperiment, DeepLearnableExperiment):
    pass


class DrakeMultibodyLearnableExperiment(DrakeExperiment):

    def __init__(self, config: SupervisedLearningExperimentConfig) -> None:
        super().__init__(config)
        self.learnable_config = cast(MultibodyLearnableSystemConfig,
                                self.config.learnable_config)
        if self.learnable_config.loss == MultibodyLosses.CONTACTNETS_LOSS:
            self.loss_callback = self.contactnets_loss
        elif self.learnable_config.loss == MultibodyLosses.PREDICTION_LOSS:
            self.loss_callback = self.prediction_with_regularization_loss
        else:
            raise RuntimeError(f"Loss {self.learnable_config.loss} not " + \
                               f"recognized for Drake multibody experiment.")

    def get_learned_system(self, _: Tensor) -> MultibodyLearnableSystem:
        learnable_config = cast(MultibodyLearnableSystemConfig,
                                self.config.learnable_config)
        output_dir = file_utils.get_learned_urdf_dir(self.config.storage,
                                                     self.config.run_name)
        return MultibodyLearnableSystem(
            learnable_config.urdfs,
            self.config.data_config.dt,
            learnable_config.inertia_mode,
            learnable_config.loss_variation,
            w_pred = learnable_config.w_pred,
            w_comp = learnable_config.w_comp.value,
            w_diss = learnable_config.w_diss.value,
            w_pen = learnable_config.w_pen.value,
            w_res = learnable_config.w_res.value,
            w_res_w = learnable_config.w_res_w.value,
            output_urdfs_dir=output_dir,
            do_residual=learnable_config.do_residual,
            represent_geometry_as=learnable_config.represent_geometry_as,
            randomize_initialization=learnable_config.randomize_initialization,
            g_frac=learnable_config.g_frac)

    def write_to_wandb(self, epoch: int, learned_system: System,
                       statistics: Dict) -> None:
        """In addition to extracting and writing training progress summary via
        the parent :py:meth:`Experiment.write_to_wandb` method, also make a
        breakdown plot of loss contributions for the ContactNets loss
        formulation.

        Args:
            epoch: Current epoch.
            learned_system: System being trained.
            statistics: Summary statistics for learning process.
        """
        assert self.wandb_manager is not None

        # begin recording wall-clock logging time.
        start_log_time = time.time()

        # To save space on W&B storage, only generate comparison videos at first
        # and best epoch, the latter of which is implemented in
        # :meth:`_evaluation`.
        skip_videos = False if epoch==0 else True

        epoch_vars, learned_system_summary = \
            self.build_epoch_vars_and_system_summary(statistics, learned_system,
                                                     skip_videos=skip_videos)

        # Start computing individual loss components.
        # First get a batch sized portion of the shuffled training set.
        train_traj_set, _, _ = \
            self.learning_data_manager.get_updated_trajectory_sets()
        train_dataloader = DataLoader(
            train_traj_set.slices,
            batch_size=self.config.optimizer_config.batch_size.value,
            shuffle=True)

        # Calculate the average loss components.
        losses_pred, losses_comp, losses_pen, losses_diss = [], [], [], []
        residual_norm, residual_weight, inertia_cond_num = [], [], []
        for xy_i in train_dataloader:
            x_i: Tensor = xy_i[0]
            y_i: Tensor = xy_i[1]

            x = x_i[..., -1, :]
            x_plus = y_i[..., 0, :]
            u = torch.zeros(x.shape[:-1] + (0,))

            loss_pred, loss_comp, loss_pen, loss_diss = \
                learned_system.calculate_contactnets_loss_terms(x, u, x_plus)
            regularizers = \
                learned_system.get_regularization_terms(x, u, x_plus)

            losses_pred.append(loss_pred.clone().detach())
            losses_comp.append(loss_comp.clone().detach())
            losses_pen.append(loss_pen.clone().detach())
            losses_diss.append(loss_diss.clone().detach())
            residual_norm.append(regularizers[0].clone().detach())
            residual_weight.append(regularizers[1].clone().detach())
            inertia_cond_num.append(regularizers[2].clone().detach())

        def really_weird_fix_for_cluster_only(list_of_tensors):
            """For some reason, on the cluster only, the last item in the loss
            lists can be a different shape than the rest of the items, and this
            results in an error with the ``sum(losses_pred)`` below.  For now,
            the fix (hack) is to just drop that last term.

            TODO:  Figure out what is going on.
            """
            if (len(list_of_tensors) > 1) and \
               (list_of_tensors[-1].shape != list_of_tensors[0].shape):
                    return list_of_tensors[:-1]
            return list_of_tensors

        losses_pred = really_weird_fix_for_cluster_only(losses_pred)
        losses_comp = really_weird_fix_for_cluster_only(losses_comp)
        losses_pen = really_weird_fix_for_cluster_only(losses_pen)
        losses_diss = really_weird_fix_for_cluster_only(losses_diss)
        residual_norm = really_weird_fix_for_cluster_only(residual_norm)
        residual_weight = really_weird_fix_for_cluster_only(residual_weight)
        inertia_cond_num = really_weird_fix_for_cluster_only(inertia_cond_num)

        # Calculate average and scale by hyperparameter weights.
        w_pred = self.learnable_config.w_pred
        w_comp = self.learnable_config.w_comp.value
        w_diss = self.learnable_config.w_diss.value
        w_pen = self.learnable_config.w_pen.value
        w_res = self.learnable_config.w_res.value
        w_res_w = self.learnable_config.w_res_w.value

        avg_loss_pred = w_pred*cast(Tensor, sum(losses_pred) \
                            / len(losses_pred)).mean()
        avg_loss_comp = w_comp*cast(Tensor, sum(losses_comp) \
                            / len(losses_comp)).mean()
        avg_loss_pen = w_pen*cast(Tensor, sum(losses_pen) \
                            / len(losses_pen)).mean()
        avg_loss_diss = w_diss*cast(Tensor, sum(losses_diss) \
                            / len(losses_diss)).mean()
        avg_residual_norm = w_res*cast(Tensor, sum(residual_norm) \
                            / len(residual_norm)).mean()
        avg_residual_weight = w_res*cast(Tensor, sum(residual_weight) \
                            / len(residual_weight)).mean()
        avg_inertia_cond_num = 1e-5 * cast(Tensor, sum(inertia_cond_num) \
                            / len(inertia_cond_num)).mean()

        avg_loss_total = torch.sum(avg_loss_pred + avg_loss_comp + \
                                   avg_loss_pen + avg_loss_diss + \
                                   avg_residual_norm + avg_residual_weight + \
                                   avg_inertia_cond_num)

        loss_breakdown = {'loss_total': avg_loss_total,
                          'loss_pred': avg_loss_pred,
                          'loss_comp': avg_loss_comp,
                          'loss_pen': avg_loss_pen,
                          'loss_diss': avg_loss_diss,
                          'loss_res_norm': avg_residual_norm,
                          'loss_res_weight': avg_residual_weight,
                          'loss_inertia_cond': avg_inertia_cond_num}

        # Include the loss components into system summary.
        epoch_vars.update(loss_breakdown)
        
        # Overwrite the logging time.
        logging_duration = time.time() - start_log_time
        epoch_vars[LOGGING_DURATION] = logging_duration

        self.wandb_manager.update(epoch, epoch_vars,
                                  learned_system_summary.videos,
                                  learned_system_summary.meshes)

    def visualizer_regeneration_is_required(self) -> bool:
        return cast(SupervisedLearningExperimentConfig,
                    self.config).update_geometry_in_videos

    def get_learned_drake_system(
            self, learned_system: System) -> Optional[DrakeSystem]:
        if self.visualizer_regeneration_is_required():
            new_urdfs = cast(MultibodyLearnableSystem,
                             learned_system).generate_updated_urdfs('vis')
            return DrakeSystem(new_urdfs, self.get_drake_system().dt,
                               g_frac=self.config.learnable_config.g_frac)
        return None

    def prediction_with_regularization_loss(
        self, x_past: Tensor, x_future: Tensor, system: System,
        keep_batch: bool = False) -> Tensor:
        """Returns prediction loss with possibly some regularization terms,
        e.g., regularization on the size/weights of a residual network, if there
        is one.
        """
        w_res = self.learnable_config.w_res.value
        w_res_w = self.learnable_config.w_res_w.value

        prediction_loss = self.prediction_loss(x_past, x_future, system,
                                               keep_batch)

        x = x_past[..., -1, :]
        u = torch.zeros(x.shape[:-1] + (0,))
        x_plus = x_future[..., 0, :]

        regularizers = system.get_regularization_terms(x, u, x_plus)
        if len(regularizers) > 3:
            assert NotImplementedError(
                "Don't recognize more than three regularization terms.")
        elif len(regularizers) == 3:
            reg_term = (regularizers[0] * w_res) + \
                       (regularizers[1] * w_res_w) + \
                       (regularizers[2] * 1e-5)
        else:
            reg_term = torch.zeros_like(prediction_loss)

        if not keep_batch:
            prediction_loss = prediction_loss.mean()
            reg_term = reg_term.mean()

        return prediction_loss + reg_term

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
