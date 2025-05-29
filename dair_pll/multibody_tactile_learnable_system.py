"""Construction and analysis of learnable multibody systems.

Similar to Drake, multibody systems are instantiated as a child class of
:py:class:`System`: :py:class:`MultibodyTactileLearnableSystem`. This object is a thin
wrapper for a :py:class:`MultibodyTerms` member variable, which manages
computation of lumped terms necessary for simulation and evaluation.

'Tactile' means that the system is supervised with contact forces / normals directly.
See whitepaper for loss terms: https://www.overleaf.com/project/682b4c612359b4208bf974dd

Simulation is implemented via Anitescu's [1] convex method.

A large portion of the internal implementation of :py:class:`DrakeSystem` is
implemented in :py:class:`MultibodyPlantDiagram`.

[1] M. Anitescu, “Optimization-based simulation of nonsmooth rigid
multibody dynamics,” Mathematical Programming, 2006,
https://doi.org/10.1007/s10107-005-0590-7
"""

from dataclasses import dataclass
from enum import Enum
from os import path
from typing import override, Optional

import gin
import torch
from torch import Tensor
from torch.nn import Module

from dair_pll import urdf_utils, file_utils
from dair_pll.learnable_trajectory import LearnableTrajectories
from dair_pll.multibody_terms import MultibodyTerms, LearnableBodySettings
from dair_pll.solvers import DynamicCvxpyLCQPLayer
from dair_pll.state_space import StateSpace, ProductSpace
#from dair_pll.tensor_utils import pbmm, broadcast_lorentz


@gin.constants_from_enum
class ModelType(Enum):
    """Type for each URDF model"""
    NONE = 0
    CONTROLLABLE = 1
    LEARNABLE = 2

@gin.constants_from_enum
class LossFunction(Enum):
    """Which Loss Function to Use"""
    NIMP = 0
    VIMP = 1

@gin.configurable("TactileHyperparameters")
@dataclass
class MultibodyTactileHyperparameters:
    """Class to specify hyperparameters"""
    loss_fn: LossFunction = field(default_factory=lambda: LossFunction.VIMP)



@gin.configurable("TactileSystem")
class MultibodyLearnableTactileSystem(Module):
    """:py:class:`System` interface for dynamics associated with
    :py:class:`MultibodyTerms` and Tactile Supervision."""

    _multibody_terms: MultibodyTerms
    r"""Underlying drake multibody computations object"""
    _learned_trajectory: LearnableTrajectories
    r"""The learnable trajectory for the given models"""
    _controlled_space: StateSpace
    r"""The StateSpace that can be controlled."""

    _solver: DynamicCvxpyLCQPLayer
    r"""SOC QP Solver"""
    _hyperparameters: MultibodyTactileHyperparameters
    r"""Hyperparameter object"""

    def __init__(
        self,
        urdfs: dict[ModelType, dict[str, str]],
        learnable_body_dict: Optional[dict[str, LearnableBodySettings]] = None,
        hyperparameters: MultibodyTactileHyperparameters = MultibodyTactileHyperparameters(),
        init_learned_state: Optional[list[float] | Tensor] = None,
    ) -> None:
        """Inits :py:class:`MultibodyLearnableTactileSystem` with provided model URDFs.

        Implementation is primarily based on Drake. Bodies are modeled via
        :py:class:`MultibodyTerms`, which uses Drake symbolics to generate
        dynamics terms, and the system can be exported back to a
        Drake-interpretable representation as a set of URDFs.

        Args:
            urdfs: For each model type: names and corresponding URDFs to model with
              :py:class:`MultibodyTerms`.
            learnable_body_dict: dict of body names and which properties should
              be learned
            hyperparameters: Learning hyperparameters
            init_learned_state: initial state for the learnable models, (learn_spaces.n_x,)
        """
        super().__init__()

        # Input Validation
        if learnable_body_dict is None:
            learnable_body_dict = {}

        # Init Multibody Terms
        urdf_dict = {}
        for val in urdfs.values():
            urdf_dict.update(val)
        multibody_terms = MultibodyTerms(
            urdf_dict,
            learnable_body_dict,
        )
        self._multibody_terms = multibody_terms

        # TODO: HACK re-add random initialization

        # Fill other class attributes
        # Pylint doesn't know about gin
        # pylint: disable=no-value-for-parameter
        self._solver = DynamicCvxpyLCQPLayer()
        self._hyperparameters = hyperparameters

        ## Populate Model Spaces
        learn_spaces = []
        ctrl_spaces = []
        for model_id, space in zip(
            multibody_terms.plant_diagram.model_ids, multibody_terms.plant_diagram.space.spaces
        ):
            name = multibody_terms.plant_diagram.plant.GetModelInstanceName(model_id)
            if name in urdfs[ModelType.LEARNABLE]:
                learn_spaces.append(space)
            elif name in urdfs[ModelType.CONTROLLABLE]:
                ctrl_spaces.append(space)
        self._controlled_space = ProductSpace(ctrl_spaces)

        ## Create Learnable Trajectory
        init_state = None
        if init_learned_state is not None:
            init_state = (
                init_learned_state
                if isinstance(init_learned_state, Tensor)
                else torch.tensor(init_learned_state)
            )
        self._learned_trajectory = LearnableTrajectories(ProductSpace(learn_spaces), init_state)

    def generate_updated_urdfs(
        self, output_directory: str, suffix: str = None
    ) -> dict[str, str]:
        """Exports current parameterization as a map of model_name : URDF.
        Writes URDF and all sub-meshes to provided output directory.

        Returns:
            Map of model_name : URDF string
        """
        assert output_directory is not None
        new_urdf_strings = urdf_utils.represent_multibody_terms_as_urdfs(
            self.multibody_terms, output_directory
        )

        # saves new urdfs with model name plus optional suffix in
        # new folder.
        for urdf_name, new_urdf_string in new_urdf_strings.items():
            new_urdf_filename = urdf_name + ".urdf"
            if suffix is not None:
                new_urdf_filename = (
                    new_urdf_filename.split(".")[0] + "_" + suffix + ".urdf"
                )

            new_urdf_path = path.join(output_directory, new_urdf_filename)
            file_utils.save_string(new_urdf_path, new_urdf_string)

        return new_urdf_strings

    def add_trajectories(
        self, traj_lens: List[int], traj_data: Optional[List[Optional[Tensor]]] = None
    ):
        """Extend learnable trajectory by length sum(traj_lens) with data in traj_data"""
        if traj_data is None:
            traj_data = [None] * len(traj_lens)
        assert len(traj_data) == len(traj_lens)
        for traj_len, traj_datum in zip(traj_lens, traj_data):
            self._trajectory.add_trajectory(traj_len, traj_datum)

    def forward(
        self, ctrl_traj: Tensor, timestamps: Tensor, nimp_override: bool = False
    ) -> tuple[Tensor, Tensor, ]:
        """Forward Function for the Learnable System

        If NIMP (or nimp_override), use diffsim to get system state. Otherwise concat
        from learnable trajectory. 

        Args:
            ctrl_traj: Input robot trajectory of size (batch, traj_len, self._controlled_space.n_x)
            timestamps: (batch, traj_len)
            nimp_override: If true, run diffsim from current object pose.

        Returns:
            system_state (batch, traj_len, self.multibody_terms.plant_diagram.space.n_x)
            phi (batch, traj_len, n_object_pairs, n_contacts)
            normals (batch, traj_len, n_object_pairs, n_contacts, 3) 
            (nimp only) lambda (batch, traj_len, n_object_pairs, n_contacts, 3) 
        """
