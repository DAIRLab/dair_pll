"""Wrappers for Drake/ContactNets multibody experiments."""
from abc import ABC
from dataclasses import field, dataclass
from enum import Enum
from typing import Optional, cast, Dict, List

import torch
from torch import Tensor

from dair_pll import file_utils
from dair_pll import vis_utils
from dair_pll.deep_learnable_system import DeepLearnableExperiment
from dair_pll.drake_system import DrakeSystem
from dair_pll.experiment import SupervisedLearningExperiment, \
    LossCallbackCallable
from dair_pll.experiment_config import SystemConfig, \
    SupervisedLearningExperimentConfig
from dair_pll.geometry import MeshRepresentation, GeometryRelativeErrorFactory
from dair_pll.multibody_learnable_system import \
    MultibodyLearnableSystem
from dair_pll.state_space import partial_sum_batch
from dair_pll.summary_statistics_constants import LEARNED_SYSTEM_NAME, \
    TARGET_NAME, PREDICTION_NAME
from dair_pll.system import System, SystemSummary
from dair_pll.tensor_utils import pbmm

PARAMETER_RELATIVE_ERROR = 'parameter_relative_error'
GEOMETRY_PREFIX = 'geometry'
FRICTION_PREFIX = 'friction'


@dataclass
class DrakeSystemConfig(SystemConfig):
    urdfs: Dict[str, str] = field(default_factory=dict)


class MultibodyLosses(Enum):
    PREDICTION_LOSS = 1
    CONTACTNETS_NCP_LOSS = 2
    CONTACTNETS_ANITESCU_LOSS = 3


@dataclass
class MultibodyLearnableSystemConfig(DrakeSystemConfig):
    learn_inertial_parameters: bool = False
    """Relative noise to add to parameter initial conditions."""
    mesh_representation: MeshRepresentation = MeshRepresentation.POLYGON
    """Whether meshes are represented as polygons or deep networks."""
    initial_parameter_noise_level: Tensor = torch.tensor(0.)
    """Relative noise level to add to initial parameters."""

    def __post_init__(self) -> None:
        if self.learn_inertial_parameters:
            raise NotImplementedError(
                'Learning inertial parameters is not yet supported, '
                'as randomized initial conditions are not yet implemented.')


@dataclass
class DrakeMultibodyLearnableExperimentConfig(SupervisedLearningExperimentConfig
                                             ):
    visualize_learned_geometry: bool = True
    """Whether to use learned geometry in trajectory overlay visualization."""
    training_loss: MultibodyLosses = MultibodyLosses.PREDICTION_LOSS
    """Whether to use ContactNets or prediction loss for training."""
    contactnets_length_scale: float = 1.0
    """How much to weight velocities vs. positions in ContactNets loss."""


class DrakeExperiment(SupervisedLearningExperiment, ABC):
    base_drake_system: Optional[DrakeSystem]
    visualization_system: Optional[DrakeSystem]
    oracle_multibody_learnable_system: Optional[MultibodyLearnableSystem]

    def prediction_loss(self,
                        x_past: Tensor,
                        x_future: Tensor,
                        system: System,
                        keep_batch: bool = False) -> Tensor:
        r"""Default :py:data:`LossCallbackCallable` which evaluates to system's
        :math:`l_2` prediction error on batch:

        .. math::

            \mathcal{L}(x_{p,i,\cdot}, x_{f,i,\cdot}) = \sum_{j} ||\hat x_{f,
            i,j} - x_{f,i,j}||^2,

        where :math:`x_{p,i,\cdot}, x_{f,i,\cdot}` are the :math:`i`\ th
        elements of the past and future batches; and
        :math:`\hat x_{f,i,j}` is the :math:`j`-step forward prediction of the
        model from the past batch.

        See :py:data:`LossCallbackCallable` for additional type signature info.
        """
        space = self.space
        x_predicted = self.batch_predict(x_past, system)
        v_future = space.v(x_future)
        v_predicted = space.v(x_predicted)
        avg_const = v_predicted.nelement() // v_predicted.shape[0]
        # scale by mass
        q_future = space.q(x_future)
        mass = self.get_oracle_multibody_learnable_system(
        ).multibody_terms.lagrangian_terms.get_mass_matrix(q_future)
        if not keep_batch:
            avg_const *= x_predicted.shape[0]
        velocity_error = (v_predicted - v_future).unsqueeze(-1)
        energetic_error = pbmm(velocity_error.mT,
                               pbmm(mass, velocity_error)).squeeze(-1)
        return partial_sum_batch(energetic_error, keep_batch) / avg_const

    def __init__(self, config: SupervisedLearningExperimentConfig) -> None:
        super().__init__(config)
        self.base_drake_system = None
        self.visualization_system = None
        self.oracle_multibody_learnable_system = None
        self.loss_callback = cast(LossCallbackCallable, self.prediction_loss)

    def get_drake_system(self) -> DrakeSystem:
        has_property = hasattr(self, 'base_drake_system')
        if not has_property or self.base_drake_system is None:
            base_config = cast(DrakeSystemConfig, self.config.base_config)
            dt = self.config.data_config.dt
            self.base_drake_system = DrakeSystem(base_config.urdfs, dt)
        return self.base_drake_system

    def get_multibody_learnable_system_from_drake_config(
            self, drake_config: DrakeSystemConfig) -> MultibodyLearnableSystem:
        learn_inertial_parameters = False
        parameter_noise_level = torch.tensor(0.0)
        mesh_representation = MeshRepresentation.POLYGON

        if isinstance(drake_config, MultibodyLearnableSystemConfig):
            learn_inertial_parameters = drake_config.learn_inertial_parameters
            parameter_noise_level = drake_config.initial_parameter_noise_level
            mesh_representation = drake_config.mesh_representation

        output_dir = file_utils.get_learned_urdf_dir(self.config.storage,
                                                     self.config.run_name)

        return MultibodyLearnableSystem(
            drake_config.urdfs,
            self.config.data_config.dt,
            output_urdfs_dir=output_dir,
            learn_inertial_parameters=learn_inertial_parameters,
            mesh_representation=mesh_representation,
            parameter_noise_level=parameter_noise_level)

    def get_oracle_multibody_learnable_system(self) -> MultibodyLearnableSystem:
        has_property = hasattr(self, 'oracle_multibody_learnable_system')
        if not has_property or self.oracle_multibody_learnable_system is None:
            self.oracle_multibody_learnable_system = \
                self.get_multibody_learnable_system_from_drake_config(
                    cast(DrakeSystemConfig, self.config.base_config))
        return self.oracle_multibody_learnable_system

    def get_oracle_system(self) -> System:
        return self.get_oracle_multibody_learnable_system()

    def get_base_system(self) -> System:
        return self.get_drake_system()

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
        return SystemSummary(scalars={}, videos=videos, meshes={})


class DrakeDeepLearnableExperiment(DrakeExperiment, DeepLearnableExperiment):
    pass


class DrakeMultibodyLearnableExperiment(DrakeExperiment):

    def __init__(self, config: DrakeMultibodyLearnableExperimentConfig) -> None:
        super().__init__(config)
        if config.training_loss != MultibodyLosses.PREDICTION_LOSS:
            self.loss_callback = self.contactnets_loss

    def get_learned_system(self, _: Tensor) -> MultibodyLearnableSystem:
        system = self.get_multibody_learnable_system_from_drake_config(
            cast(MultibodyLearnableSystemConfig, self.config.learnable_config))
        system.contactnets_length_scale = cast(
            DrakeMultibodyLearnableExperimentConfig,
            self.config).contactnets_length_scale
        return system

    def visualizer_regeneration_is_required(self) -> bool:
        return cast(DrakeMultibodyLearnableExperimentConfig,
                    self.config).visualize_learned_geometry

    def get_learned_drake_system(
            self, learned_system: System) -> Optional[DrakeSystem]:
        visualize_learned_geometry = cast(
            DrakeMultibodyLearnableExperimentConfig,
            self.config).visualize_learned_geometry

        if visualize_learned_geometry:
            new_urdfs = cast(MultibodyLearnableSystem,
                             learned_system).generate_updated_urdfs()
            return DrakeSystem(new_urdfs, self.get_drake_system().dt)
        return None

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
        training_loss = cast(DrakeMultibodyLearnableExperimentConfig,
                             self.config).training_loss
        if training_loss == MultibodyLosses.CONTACTNETS_NCP_LOSS:
            loss = system.contactnets_loss_ncp(x, u, x_plus)
        else:
            loss = system.contactnets_loss_anitescu(x, u, x_plus)
        if not keep_batch:
            loss = loss.mean()
        return loss

    def base_and_learned_comparison_summary(
            self, statistics: Dict, learned_system: System) -> SystemSummary:
        r"""Extracts a :py:class:`~dair_pll.system.SystemSummary` that compares
        the base system to the learned system.

        This function extends the visualization tools provided in
        :py:meth:`DrakeExperiment.base_and_learned_comparison_summary` with a
        calculation of relative parameter error.

        Args:
            statistics: Dictionary of training statistics.
            learned_system: Most updated version of learned system during
              training.

        Returns:
            Summary containing overlaid video(s).
        """
        learnable_config = cast(MultibodyLearnableSystemConfig,
                                self.config.learnable_config)
        if learnable_config.learn_inertial_parameters:
            raise NotImplementedError('TODO: implement relative error for '
                                      'inertial parameters')

        summary = super().base_and_learned_comparison_summary(
            statistics, learned_system)

        learned_system = cast(MultibodyLearnableSystem, learned_system)
        oracle_system = self.get_oracle_multibody_learnable_system()

        # friction
        mu_learned = learned_system.multibody_terms.contact_terms. \
            get_lumped_friction_coefficients()
        mu_true = oracle_system.multibody_terms.contact_terms. \
            get_lumped_friction_coefficients()
        friction_relative_errors = [(mu_l - mu_t).abs().unsqueeze(0) / mu_t
                                    for (mu_l,
                                         mu_t) in zip(mu_learned, mu_true)]

        # geometry
        geometries_learned = \
            learned_system.multibody_terms.contact_terms.geometries
        geometries_true = \
            oracle_system.multibody_terms.contact_terms.geometries

        geometry_relative_errors = cast(List[Tensor], [])
        for geometry_learned, geometry_true in zip(geometries_learned,
                                                   geometries_true):
            geometry_pair_error = \
                GeometryRelativeErrorFactory.calculate_error(
                    geometry_learned, geometry_true)

            if geometry_pair_error is not None:
                geometry_relative_errors.append(geometry_pair_error)

        summary.scalars[f'{FRICTION_PREFIX}_{PARAMETER_RELATIVE_ERROR}'] = \
            torch.cat(friction_relative_errors).mean().item()

        summary.scalars[f'{GEOMETRY_PREFIX}_{PARAMETER_RELATIVE_ERROR}'] = \
            torch.cat(geometry_relative_errors).mean().item()

        summary.scalars[PARAMETER_RELATIVE_ERROR] = \
            torch.cat(friction_relative_errors +
                      geometry_relative_errors).mean().item()

        return summary
