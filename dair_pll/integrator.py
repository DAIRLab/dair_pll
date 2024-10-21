"""Classes for time integration on state spaces

This module implements an :py:class:`Integrator` abstract type which is a
convenience wrapper for various forms of integrating dynamics in time over a
``StateSpace``. As state spaces are assumed to be the product space of a Lie
group and its associated algebra (G x g), there are several options for which a
"partial step" ``partial_step`` might be defined for mapping current states
to next states:

    * x -> x' in G x g (current state to next state)
    * x -> dx in g x R^n_v, and x' = x * exp(dx) (current state to state delta)
    * x -> v' in g, and q' = q * exp(v') (current state to next velocity)
    * x -> dv in R^n_v, v' = v + dv (current state to velocity delta)
    * x -> q' in G, and v' = log(q' * inv(q))/dt (current state to next
      configuration)
    * x -> dq in g, q' = q * exp(dq), v' = log(q' * inv(q))/dt (current state to
      configuration delta).

Each option is implemented as a convenience class inheriting from
:py:class:`Integrator`.

In addition to this state mapping, the integrator allows for an additional
hidden state denoted as ``carry`` to be propagated through .

:py:class:`Integrator` objects have a simulation interface that requires an
initial condition in the form of an initial state and ``carry``.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

from torch import Tensor
from torch.nn import Module

from dair_pll import tensor_utils
from dair_pll.state_space import StateSpace

PartialStepCallback = Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]


class Integrator(ABC, Module):
    """Class that manages integration in time of given dynamics.

    Takes in a ``partial_step`` callback object which defines the underlying
    dynamics abstractly. Inheriting classes makes this relationship concrete.

    This class is primarily used for its :py:meth:`simulate` method which
    integrates forward in time from a given initial condition.
    """

    partial_step_callback: Optional[PartialStepCallback]
    space: StateSpace
    dt: float
    out_size: int

    def __init__(
        self, space: StateSpace, partial_step_callback: PartialStepCallback, dt: float
    ) -> None:
        """Inits :py:class:`Integrator` with specified dynamics.

        Args:
            space: state space of system to be integrated.
            partial_step_callback: Dynamics defined as partial update from
              current state to next state. Exact usage is abstract.
            dt: time step.
        """
        super().__init__()
        self.partial_step_callback = partial_step_callback
        self.space = space
        self.dt = dt
        self.out_size = type(self).calc_out_size(space)

    def partial_step(self, x: Tensor, carry: Tensor) -> Tuple[Tensor, Tensor]:
        """Wrapper method for calling ``partial_step_callback``"""
        assert self.partial_step_callback is not None
        return self.partial_step_callback(x, carry)

    def simulate(
        self, x_0: Tensor, carry_0: Tensor, steps: int
    ) -> Tuple[Tensor, Tensor]:
        """Simulates forward in time from initial condition.

        Args:
            x_0: (\*, space.n_x) batch of initial condition states
            carry_0: (\*, ?) batch of initial condition hidden states
            steps: number of steps to simulate forward in time (>= 0)

        Returns:
            (\*, space.n_x, steps + 1) simulated trajectory.
        """
        assert steps >= 0
        assert x_0.shape[-1] == self.space.n_x
        x_trajectory = tensor_utils.tile_penultimate_dim(x_0.unsqueeze(-2), steps + 1)
        carry_trajectory = tensor_utils.tile_penultimate_dim(
            carry_0.unsqueeze(-2), steps + 1
        )
        x = x_0
        carry = carry_0
        for step in range(steps):
            x, carry = self.step(x, carry)
            x_trajectory[..., step + 1, :] = x
            carry_trajectory[..., step + 1, :] = carry
        return x_trajectory, carry_trajectory

    @abstractmethod
    def step(self, x: Tensor, carry: Tensor) -> Tuple[Tensor, Tensor]:
        """Takes single step in time.

        Abstract wrapper which inheriting classes incorporate
        :py:meth:`partial_step` into to complete a single time step.

        Args:
            x: (\*, space.n_x) current state
            carry: (\*, ?) current hidden state

        Returns:
            (\*, space.n_x) next state
            (\*, ?) next hidden state
        """

    @staticmethod
    def calc_out_size(space: StateSpace) -> int:
        """Final dimension of output shape of :py:meth:`partial_step`"""
        return space.n_x


class StateIntegrator(Integrator):
    """Convenience class for :py:class:`Integrator` where
    :py:meth:`partial_step` maps current state directly to next state."""

    def step(self, x: Tensor, carry: Tensor) -> Tuple[Tensor, Tensor]:
        """Integrates by direct passthrough to :py:meth:`partial_step`"""
        x_next, carry = self.partial_step(x, carry)
        return self.space.project_state(x_next), carry


class DeltaStateIntegrator(Integrator):
    """Convenience class for :py:class:`Integrator` where
    :py:meth:`partial_step` maps current state to state delta."""

    def step(self, x: Tensor, carry: Tensor) -> Tuple[Tensor, Tensor]:
        """Integrates by perturbing current state by output of
        :py:meth:`partial_step`"""
        space = self.space
        dx, carry = self.partial_step(x, carry)
        return space.shift_state(x, dx), carry

    @staticmethod
    def calc_out_size(space: StateSpace) -> int:
        return 2 * space.n_v


class VelocityIntegrator(Integrator):
    """Convenience class for :py:class:`Integrator` where
    :py:meth:`partial_step` maps current state to next velocity."""

    def step(self, x: Tensor, carry: Tensor) -> Tuple[Tensor, Tensor]:
        """Integrates by setting next velocity to output of
        :py:meth:`partial_step` and implicit Euler integration of the
        configuration."""
        space = self.space
        q = space.q(x)
        v_next, carry = self.partial_step(x, carry)
        q_next = space.euler_step(q, v_next, self.dt)

        return space.x(q_next, v_next), carry

    @staticmethod
    def calc_out_size(space: StateSpace) -> int:
        return space.n_v


class DeltaVelocityIntegrator(Integrator):
    """Convenience class for :py:class:`Integrator` where
    :py:meth:`partial_step` maps current state to velocity delta."""

    def step(self, x: Tensor, carry: Tensor) -> Tuple[Tensor, Tensor]:
        """Integrates by perturbing current velocity by output of
        :py:meth:`partial_step` and implicit Euler integration of the
        configuration."""
        space = self.space
        q, v = space.q_v(x)
        dv, carry = self.partial_step(x, carry)
        v_next = v + dv
        q_next = space.euler_step(q, v_next, self.dt)

        return space.x(q_next, v_next), carry

    @staticmethod
    def calc_out_size(space: StateSpace) -> int:
        return space.n_v


class ConfigurationIntegrator(Integrator):
    """Convenience class for :py:class:`Integrator` where
    :py:meth:`partial_step` maps current state to next configuration."""

    def step(self, x: Tensor, carry: Tensor) -> Tuple[Tensor, Tensor]:
        """Integrates by setting next configuration to output of
        :py:meth:`partial_step` and finite differencing for the next
        velocity."""
        space = self.space
        q = space.q(x)
        q_next_pre_projection, carry = self.partial_step(x, carry)
        q_next = space.project_configuration(q_next_pre_projection)
        v_next = space.finite_difference(q, q_next, self.dt)
        return space.x(q_next, v_next), carry

    @staticmethod
    def calc_out_size(space: StateSpace) -> int:
        return space.n_q


class DeltaConfigurationIntegrator(Integrator):
    """Convenience class for :py:class:`Integrator` where
    :py:meth:`partial_step` maps current state to configuration delta."""

    def step(self, x: Tensor, carry: Tensor) -> Tuple[Tensor, Tensor]:
        """Integrates by perturbing current configuration by output of
        :py:meth:`partial_step` and finite differencing for the next
        velocity."""
        space = self.space
        q = space.q(x)
        dq, carry = self.partial_step(x, carry)
        q_next = space.exponential(q, dq)
        v_next = dq / self.dt
        return space.x(q_next, v_next), carry

    @staticmethod
    def calc_out_size(space: StateSpace) -> int:
        return space.n_v
