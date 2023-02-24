"""System abstract type definition.

This file contains the fundamental ``System`` interface for dynamical
systems. Systems are defined by their underlying dynamics; integration
scheme; and an initial condition sampler for both states and hidden
states/``carry``.

Unlike ``Integrator``, ``System`` requires a temporal sequence of initial
condition states; this is done to accommodate systems with hidden states that
behave differently when "preloaded" with an initialization trajectory,
such as a UKF estimator or an RNN.

``System`` is used to interface with external simulators, e.g. Drake and MuJoCo.
"""
from abc import ABC
from dataclasses import dataclass, field
from typing import Tuple, Callable, Optional, Dict, List

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from dair_pll import state_space
from dair_pll.integrator import Integrator
from dair_pll.state_space import StateSpace, StateSpaceSampler


@dataclass
class MeshSummary:
    r""":py:func:`dataclasses.dataclass` for mesh visualization."""
    vertices: Tensor = Tensor()
    r"""Vertices in mesh, ``(n_vert, 3)``\ ."""
    faces: Tensor = Tensor()
    r"""3-tuple indices of vertices that form faces, ``(n_face, 3)``\ ."""


@dataclass
class SystemSummary:
    """:py:func:`dataclasses.dataclass` for reporting information about the
    progress of a training run."""
    scalars: Dict[str, float] = field(default_factory=dict)
    videos: Dict[str, Tuple[np.ndarray, int]] = field(default_factory=dict)
    meshes: Dict[str, MeshSummary] = field(default_factory=dict)
    overlaid_scalars: Optional[List[Dict[str, float]]] = None


class System(ABC, Module):
    """Class for encapsulating a dynamical system.

    Primarily implemented as a thin shell of ``Integrator`` with various
    sampling interfaces defined.

    A major difference from the ``Integrator`` interface is that ``System``
    accepts a sequence of states, along with a single ``carry``/hidden state, as
    an initial condition to accommodate proper initialization of some types
    of recurrent dynamics.
    """
    space: StateSpace
    integrator: Integrator
    state_sampler: StateSpaceSampler
    carry_callback: Optional[Callable[[], Tensor]]
    max_batch_dim: Optional[int]

    def __init__(self,
                 space: StateSpace,
                 integrator: Integrator,
                 max_batch_dim: Optional[int] = None) -> None:
        """Inits ``System`` with prescribed integration properties.

        Args:
            space: State space of underlying dynamics
            integrator: Integrator of underlying dynamics
            max_batch_dim: Maximum number of batch dimensions supported by
            ``integrator``.
        """
        super().__init__()
        self.space = space
        self.integrator = integrator
        self.state_sampler = state_space.ZeroSampler(space)
        # pylint: disable=E1103
        self.carry_callback = lambda: torch.zeros((1, 1))
        self.max_batch_dim = max_batch_dim

    def sample_trajectory(self, length: int) -> Tuple[Tensor, Tensor]:
        """Sample

        Args:
            length: duration of trajectory in number of time steps

        Returns:
            (length, space.nx) state trajectory
            (length, ?) carry trajectory
        """
        x_0, carry_0 = self.sample_initial_condition()
        return self.simulate(x_0, carry_0, length)

    def simulate(self,
                 x_0: Tensor,
                 carry_0: Tensor,
                 steps: int = 1) -> Tuple[Tensor, Tensor]:
        """Simulate forward in time from initial condition.

        Args:
            x_0: ``(*, T_0, space.n_x)`` initial state sequence
            carry_0: ``(*, ?)`` initial hidden state
            steps: number of steps to take beyond initial condition

        Returns:
            ``(*, steps + 1, space.n_x)`` state trajectory
            ``(*, steps + 1, ?)`` hidden state trajectory
        """

        # If batching is more dimensions than allowed, iterate over outer
        # dimension.
        if self.max_batch_dim and (x_0.dim() - 2) > self.max_batch_dim:
            x_carry_list = [
                self.simulate(x0i, c0i) for x0i, c0i in zip(x_0, carry_0)
            ]
            # pylint: disable=E1103
            x_trajectory = torch.stack([x_carry[0] for x_carry in x_carry_list])
            carry_trajectory = torch.stack(
                [x_carry[1] for x_carry in x_carry_list])
        else:
            x, carry = self.preprocess_initial_condition(x_0, carry_0)
            x_trajectory, carry_trajectory = self.integrator.simulate(
                x, carry, steps)
        return x_trajectory, carry_trajectory

    def sample_initial_condition(self) -> Tuple[Tensor, Tensor]:
        """Queries state and hidden state samplers for initial condition."""
        assert self.carry_callback is not None

        # Reshapes (space.n_x,) sample into duration-1 sequence.
        return self.state_sampler.get_sample().reshape(
            1, self.space.n_x), self.carry_callback()

    def set_state_sampler(self, sampler: StateSpaceSampler) -> None:
        """Setter for state initial condition sampler."""
        self.state_sampler = sampler

    def set_carry_sampler(self, callback: Callable[[], Tensor]) -> None:
        """Setter for hidden state initial condition sampler."""
        self.carry_callback = callback

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
        assert len(x_0.shape) >= 2
        assert len(carry_0.shape) >= 1
        if self.max_batch_dim is not None:
            assert len(x_0.shape) <= 2 + self.max_batch_dim
            assert len(carry_0.shape) <= 1 + self.max_batch_dim
        assert x_0.shape[-1] == self.space.n_x

        # Just return most recent state, don't do anything to hidden state
        return x_0[..., -1, :], carry_0

    def summary(self, statistics: Dict) -> SystemSummary:
        """Summarizes the current behavior and properties of the system.

        Args:
            statistics: dictionary of training statistics

        Returns:
            Summary of system.

        Todo:
            Update for structured statistics object.
            Fix ``pylint`` warning elegantly.
        """
        # no-ops to prevent pesky pylint errors
        assert statistics is not None
        assert self is not None
        return SystemSummary()
