"""Utility classes for interfacing with Drake's internal state format.

Classes herein mainly support the implementation of ``DrakeStateConverter``. In
order to make Drake states compatible with available ``StateSpace``
inheriting classes, users must define the drake system by a collection of
URDF files, each of which contains a model for exactly one floating- or
fixed-base rigid multibody chain. This allows for the system to be modeled as
having a ``ProductSpace`` state space, where each factor space is a
``FloatingBaseSpace`` or ``FixedBaseSpace``.

For flexible usage, the conversion is between contexts and numpy ``ndarray``
types. This is particularly useful as it allows pydrake symbolic types to
be used, facilitating differentiable geometric analysis of the relationship
between the coordinate systems in ``multibody_terms.py``.
"""
from typing import Tuple, List

import numpy as np
from pydrake.multibody.plant import MultibodyPlant  # type: ignore
from pydrake.multibody.tree import ModelInstanceIndex  # type: ignore
from pydrake.systems.framework import Context  # type: ignore

from dair_pll import quaternion
from dair_pll import state_space


def state_ndarray_reformat(x: np.ndarray) -> np.ndarray:
    """Resizes Drake coordinates to ``StateSpace`` batch."""
    return np.copy(x).reshape(1, x.size)


def drake_ndarray_reformat(x: np.ndarray) -> np.ndarray:
    """Resizes ``StateSpace`` batch to Drake coordinates."""
    return np.copy(x).reshape(x.size)


class DrakeFloatingBaseStateConverter:
    """Converts between the ``np.ndarray`` state coordinates of a Drake
    MultibodyPlant model instance and a floating-base open kinematic chain.

    When a Drake model instance is a single floating-base rigid chain,
    it represents the configuration in tangent bundle of SE(3) x R^n_joints,
    with coordinates as a quaternion; world-frame floating base
    c.o.m., joint positions/angles, world-axes floating base angular/linear
    velocity, and joint velocities.

    Conversion between coordinate sets is then simply a frame transformation
    on the angular velocity between world and floating base frame.
    """

    @staticmethod
    def drake_to_state(q_drake: np.ndarray,
                       v_drake: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Formats configuration and velocity into row vectors, and rotates
        angular velocity into body frame."""
        q = state_ndarray_reformat(q_drake)
        v = state_ndarray_reformat(v_drake)
        v[..., :3] = quaternion.rotate(quaternion.inverse(q[..., :4]),
                                        v[..., :3])
        return q, v

    @staticmethod
    def state_to_drake(q: np.ndarray,
                       v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Formats configuration and velocity into squeezed vectors,
        and rotates angular velocity into world frame."""
        q_drake = state_ndarray_reformat(q)
        v_drake = state_ndarray_reformat(v)
        v_drake[..., :3] = quaternion.rotate(q[..., :4].reshape(1, -1),
                                              v_drake[..., :3])
        q_drake = q_drake.reshape(q_drake.size)
        v_drake = v_drake.reshape(v_drake.size)
        return q_drake, v_drake


class DrakeFixedBaseStateConverter:
    """Converts between the ``np.ndarray`` state coordinates of a Drake
    MultibodyPlant model instance and a fixed-base open kinematic chain.

    When a Drake model instance is a single fixed-base rigid chain,
    it represents the configuration in tangent bundle of R^n_joints,
    with the same exact coordinate system that ``FixedBaseSpace`` uses.
    Therefore, conversion between these types is a simple passthrough that
    copies the coordinates in memory.
    """

    @staticmethod
    def drake_to_state(q_drake: np.ndarray,
                       v_drake: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Formats configuration and velocity into row vectors."""
        q = state_ndarray_reformat(q_drake)
        v = state_ndarray_reformat(v_drake)
        return q, v

    @staticmethod
    def state_to_drake(q: np.ndarray,
                       v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Formats configuration and velocity into squeezed vectors."""
        q_drake = drake_ndarray_reformat(q)
        v_drake = drake_ndarray_reformat(v)
        return q_drake, v_drake


class DrakeModelStateConverterFactory:
    """Factory class for selecting Drake-to-``BaseSpace`` coordinate
    conversions."""

    @staticmethod
    def state_to_drake(
            q: np.ndarray, v: np.ndarray,
            space: state_space.StateSpace) -> Tuple[np.ndarray, np.ndarray]:
        """Selects ``state_to_drake`` method based on presence of floating
        base from ``DrakeFloatingBaseStateConverter`` or
        ``DrakeFixedBaseStateConverter``."""
        if isinstance(space, state_space.FloatingBaseSpace):
            return DrakeFloatingBaseStateConverter.state_to_drake(q, v)
        if isinstance(space, state_space.FixedBaseSpace):
            return DrakeFixedBaseStateConverter.state_to_drake(q, v)

        raise TypeError('Argument "space" must be instance of type '
                        'FloatingBaseSpace or FixedBaseSpce!')

    @staticmethod
    def drake_to_state(
            q_drake: np.ndarray, v_drake: np.ndarray,
            space: state_space.StateSpace) -> Tuple[np.ndarray, np.ndarray]:
        """Selects ``drake_to_state`` method based on presence of floating
        base from ``DrakeFloatingBaseStateConverter`` or
        ``DrakeFixedBaseStateConverter``."""
        if isinstance(space, state_space.FloatingBaseSpace):
            return DrakeFloatingBaseStateConverter.drake_to_state(
                q_drake, v_drake)
        if isinstance(space, state_space.FixedBaseSpace):
            return DrakeFixedBaseStateConverter.drake_to_state(q_drake, v_drake)

        raise TypeError('Argument "space" must be instance of type '
                        'FloatingBaseSpace or FixedBaseSpce!')


class DrakeStateConverter:
    """Utility namespace for conversion between Drake state format and
    ``ProductSpace`` formats.

    Given a ``MultibodyPlant`` and a complete list of its models,
    ``DrakeStateConverter`` converts between a numpy ``ndarray`` and ``Context``
    representation of the state space. This class leverages the
    one-open-kinematic-chain-per-model assumption to iterate over the models,
    and convert the factor space coordinates with
    ``DrakeModelStateConverterFactory``.
    """

    @staticmethod
    def context_to_state(plant: MultibodyPlant, plant_context: Context,
                         model_ids: List[ModelInstanceIndex],
                         space: state_space.ProductSpace) -> np.ndarray:
        """Retrieves ``ProductSpace``-formatted state from plant's context.

        Args:
            plant: plant from which to retrieve state.
            plant_context: plant's context which stores its state.
            model_ids: List of plant's models
            space: state space of output state.

        Returns:
            (space.n_x,) current state of plant.
        """
        qs = []
        vs = []
        spaces = space.spaces
        for model_id, model_space in zip(model_ids, spaces):
            q_drake = plant.GetPositions(plant_context, model_id)
            v_drake = plant.GetVelocities(plant_context, model_id)
            q, v = DrakeModelStateConverterFactory.drake_to_state(
                q_drake, v_drake, model_space)
            qs.append(q)
            vs.append(v)
        q = np.concatenate(qs, axis=-1)
        v = np.concatenate(vs, axis=-1)
        return np.concatenate([q, v], axis=-1).squeeze()

    @staticmethod
    def state_to_context(plant: MultibodyPlant, plant_context: Context,
                         x: np.ndarray, model_ids: List[ModelInstanceIndex],
                         space: state_space.ProductSpace) -> None:
        """Transforms and assigns ``ProductSpace``-formatted state in plant's
        mutable context.

        Args:
            plant: plant in which to store state.
            plant_context: plant's context which stores its state.
            x: (1, space.n_x) or (space.n_x,) state.
            model_ids: Mapping from plant's model names to instances
            space: state space of output state.
        """
        assert x.shape[-1] == space.n_x
        qs = np.array_split(x[..., :space.n_q], space.q_splits, -1)
        vs = np.array_split(x[..., space.n_q:], space.v_splits, -1)
        spaces = space.spaces
        for model_id, model_space, model_q, model_v in zip(
                model_ids, spaces, qs, vs):
            (q_drake, v_drake) = DrakeModelStateConverterFactory.state_to_drake(
                model_q, model_v, model_space)

            plant.SetPositions(plant_context, model_id, q_drake)
            plant.SetVelocities(plant_context, model_id, v_drake)
