#!/usr/bin/env python3

"""Construction and use of a learnable trajectory.
"""

import bisect
import enum
from typing import Callable, Optional, List

import gin
import torch
from torch import Tensor
from torch.nn import Module, ParameterList, Parameter

from dair_pll.state_space import StateSpace, FloatingBaseSpace


@gin.constants_from_enum
class TrajectoryType(enum.Enum):
    """Possible interpolation schemes for trajectory"""
    FIRST_ORDER = 0
    CUBIC_HERMITE = 1


@gin.configurable(denylist=["space"])
class LearnableTrajectory(Module):
    """
    Piecewise Polynomial Learnable Trajectory
    """

    _breaks: Tensor
    _samples: Tensor
    _samples_dot: Tensor
    _samples_registered: ParameterList
    _space: StateSpace
    _type: TrajectoryType

    _interp_fn: Callable[[int, int, float], Tensor]

    def __init__(
        self,
        space: StateSpace,
        traj_type: TrajectoryType = TrajectoryType.FIRST_ORDER,
    ) -> None:
        super().__init__()
        self._space = space
        self._type = traj_type
        self._breaks = torch.tensor([])
        self._samples = []
        self._samples_registered = ParameterList([])

        self._samples_dot = None
        if traj_type == TrajectoryType.CUBIC_HERMITE:
            self._samples_dot = []
            # TODO: Add cubic interp function
            raise NotImplementedError("CubicHermite Not Implemented Yet")
        self._interp_fn = self.linear_interp

    def linear_interp(self, idx_left: Tensor, idx_right: Tensor, interp: Tensor):
        """Linear interpolation between samples, samples_dot ignored

        Args:
            idx_left: ``(batch,)`` starting break index
            idx_right: ``(batch,)`` ending break index
            interp: ``(batch,)`` floats in [0,1]

        Returns:
            ``(batch, self.space.n_x)`` tensor of interpolated states
        """
        # Get Velocity. assume velocity == 0 outside trajectory
        # If dt == 0, that implies config_diff should be zero, so just set dt finite
        dt = torch.maximum(
            self._breaks[idx_right] - self._breaks[idx_left],
            1e-8 * torch.ones_like(idx_left),
        )
        velocity = torch.div(
            self._space.configuration_difference(
                torch.vstack(self._samples)[idx_left], torch.vstack(self._samples)[idx_right]
            ),
            dt,
        )
        assert velocity.shape == interp.shape + (self._space.n_v,), str(velocity.shape)

        # Get Position, need to use exponential() since euler_step() takes float dt
        vstep = torch.mul(torch.mul(velocity, dt), interp)
        assert vstep.shape == velocity.shape
        position = self._space.exponential(torch.vstack(self._samples)[idx_left], vstep)

        return self._space.x(position, velocity)

    def empty(self) -> bool:
        """Does trajectory have any knot points"""
        return len(self._breaks) == 0

    @torch.no_grad()
    def add_breaks(
        self,
        breaks: Tensor,
        samples: Optional[Tensor] = None,
        samples_dot: Optional[Tensor] = None,
    ):
        """
        Adds breaks to the trajectory.
        If no samples are given, defaults to zero-state

        Args:
            breaks: ``(batch,)`` index of each sample
            samples: ``(batch, self._space.n_q)`` 
            samples_dot: ``(batch, self._space.n_v)`` 
        """
        assert breaks is not None
        if samples_dot is not None and self._samples_dot is None:
            print("WARNING: cannot add samples_dot to trajectory. Ignoring...")

        if samples is None:
            samples = self._space.q(self._space.zero_state()).expand(
                breaks.size() + (self._space.n_q,)
            )
        if samples_dot is None:
            samples_dot = self._space.v(self._space.zero_state()).expand(
                breaks.size() + (self._space.n_v,)
            )

        # Flatten Tensors
        breaks_flat = breaks.flatten()
        samples_flat = samples.flatten(end_dim=-2)
        samples_dot_flat = samples_dot.flatten(end_dim=-2)
        assert samples_flat.size() == breaks.size() + (self._space.n_q,)
        assert samples_dot_flat.size() == breaks.size() + (self._space.n_v,)

        # Insert parameters into trajectory
        for break_idx in range(breaks_flat.numel()):
            sample_insert = Parameter(
                torch.clone(samples_flat[break_idx, :]), requires_grad=True
            )
            sample_dot_insert = Parameter(
                torch.clone(samples_dot_flat[break_idx, :]), requires_grad=True
            )
            idx = bisect.bisect(self._breaks, breaks_flat[break_idx])
            # if already in trajectory, update, else insert
            if idx > 0 and self._breaks[idx] == breaks_flat[break_idx]:
                self._samples[idx - 1] = sample_insert
                if self._samples_dot is not None:
                    self._samples_dot[idx - 1] = sample_dot_insert
            else:
                self._breaks = torch.cat(
                    [self._breaks[:idx], breaks_flat[break_idx].reshape(1), self._breaks[idx:]], 0
                )
                self._samples.insert(idx, sample_insert)
                self._samples_registered.append(sample_insert)
                if self._samples_dot is not None:
                    self._samples_dot.insert(idx, sample_dot_insert)
                    self._samples_registered.append(sample_dot_insert)

    def forward(self, times: Tensor) -> Tensor:
        """Returns a set of interpolated trajectory points.
        times is a batch of scalars, i.e. shape [batch,]
        """
        # If empty returns zero-state
        if self.empty():
            return (
                self._space.zero_state()
                .reshape(len(times.shape) * (1,) + (self._space.n_x,))
                .expand(times.size() + (-1,))
            )

        times_flat = torch.flatten(times)

        # Get Index on either side of interp
        # bisec_left enforces left-polynomial derivative at breaks
        idx_right = torch.minimum(
            torch.searchsorted(self._breaks, times_flat),
            (len(self._breaks) - 1) * torch.ones_like(times_flat).int(),
        )
        idx_left = torch.maximum((idx_right - 1), torch.zeros_like(times_flat).int())

        # Interp Param in [0, 1], if idx_left == ide_right, denom = 0 -> div = inf, clamps to 1
        interp = torch.clamp(
            torch.div(
                (times_flat - self._breaks[idx_left]),
                (self._breaks[idx_right] - self._breaks[idx_left]),
            ),
            0.0,
            1.0,
        )

        return self._interp_fn(idx_left, idx_right, interp).reshape(
            times.size() + (self._space.n_x,)
        )


### Unit tests
if __name__ == "__main__":
    test_space = FloatingBaseSpace(n_joints=0)
    traj = LearnableTrajectory(test_space)
    
    # Test is_empty
    assert traj.empty()

    # Test Empty Interpolation
    ret = traj(torch.tensor([1.0, 2.0, 3.0]))
    assert ret.shape == (3, 7+6), str(ret) # n_x == n_q(7) + n_v(6)

    assert torch.all(ret[0, :] == test_space.zero_state())
    assert torch.all(ret[1, :] == test_space.zero_state())
    assert torch.all(ret[2, :] == test_space.zero_state())

    assert len([param for param in traj.parameters()]) == 0

    # Add single knot point
    sample0 = torch.tensor([[1., 0., 0., 0., 0., 0., 0.]])
    sample1 = torch.tensor([[1., 0., 0., 0., 0., 0., 1.]])
    sample2 = torch.tensor([[1., 0., 0., 0., 0., 0., 2.]])
    traj.add_breaks(torch.tensor([1.]), sample1)

    # 0 velocity everywhere, all at sample1
    assert torch.all(traj(torch.tensor([-0.5])) == torch.hstack((sample1, torch.tensor([[0., 0., 0., 0., 0., 0.]])))), str(traj(torch.tensor([-0.5])))
    assert torch.all(traj(torch.tensor([ 0.0])) == torch.hstack((sample1, torch.tensor([[0., 0., 0., 0., 0., 0.]])))), str(traj(torch.tensor([ 0.0])))
    assert torch.all(traj(torch.tensor([ 0.5])) == torch.hstack((sample1, torch.tensor([[0., 0., 0., 0., 0., 0.]])))), str(traj(torch.tensor([ 0.5])))
    assert torch.all(traj(torch.tensor([ 1.0])) == torch.hstack((sample1, torch.tensor([[0., 0., 0., 0., 0., 0.]])))), str(traj(torch.tensor([ 1.0])))
    assert torch.all(traj(torch.tensor([ 1.5])) == torch.hstack((sample1, torch.tensor([[0., 0., 0., 0., 0., 0.]])))), str(traj(torch.tensor([ 1.5])))
    assert torch.all(traj(torch.tensor([ 2.0])) == torch.hstack((sample1, torch.tensor([[0., 0., 0., 0., 0., 0.]])))), str(traj(torch.tensor([ 2.0])))
    assert torch.all(traj(torch.tensor([ 2.5])) == torch.hstack((sample1, torch.tensor([[0., 0., 0., 0., 0., 0.]])))), str(traj(torch.tensor([ 2.5])))

    # Added a single parameter
    assert len([param for param in traj.parameters()]) == 1


