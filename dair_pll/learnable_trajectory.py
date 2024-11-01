#!/usr/bin/env python3

"""Construction and use of a learnable trajectory.
"""
from typing import Optional

import gin
import torch
from torch import Tensor
from torch.nn import Module, ParameterList, Parameter

from dair_pll.state_space import StateSpace, FloatingBaseSpace
from dair_pll.tensor_utils import tensor_is_int


@gin.configurable(denylist=["space"])
class LearnableTrajectories(Module):
    """
    Piecewise Polynomial Learnable Trajectory
    """

    _trajectories: ParameterList
    _trajectories_x0: ParameterList
    _space: StateSpace

    def __init__(
        self,
        space: StateSpace,
        init_state: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self._space = space
        self._trajectories = ParameterList([])
        self._trajectories_x0 = ParameterList([])

        init_x0 = self._space.zero_state()
        if init_state is not None:
            assert init_state.size() == (self._space.n_x,), str(init_state)
            init_x0 = init_state.detach().clone()

        self._trajectories_x0.append(Parameter(init_x0, requires_grad=True))

    @property
    def space(self):
        """StateSpace of this trajectory"""
        return self._space

    @torch.no_grad
    def add_trajectory(
        self,
        traj_len: int,
        init_states: Optional[Tensor] = None,
    ):
        """
        Registers new trajectory parameters of length `traj_len`
        If no init_states are given, default to 0-state.

        init_states can be a single state, i.e. ``(n_x,)``
        Or every state in the trajectory can be defined, i.e. ``(traj_len, n_x)``

        Implementation Note:
        traj_i[0] maps to self._trajectories_x0[i]
        traj_i[traj_len_i-1] maps to self._trajectories_x0[i+1]
        """

        traj_idx = len(self._trajectories)
        new_x0 = self._trajectories_x0[traj_idx].clone().detach()
        new_trajectory = new_x0.clone().repeat(traj_len - 2, 1)
        next_x0 = new_x0.clone()
        if init_states is None:
            pass
        elif init_states.size() == (self._space.n_x,):
            new_x0 = init_states.clone().detach()
            new_trajectory = new_x0.clone().repeat(traj_len - 2, 1)
            next_x0 = new_x0.clone()
        elif init_states.size() == (
            traj_len,
            self._space.n_x,
        ):
            new_x0 = init_states[0, :].clone().detach()
            new_trajectory = init_states[1:-1, :].clone.detach()
            next_x0 = init_states[-1, :].clone().detach()
        else:
            raise ValueError(f"Invalid init_states size: {init_states.size()}")

        assert new_x0.size() == (self._space.n_x,), str(new_x0.size())
        assert next_x0.size() == (self._space.n_x,), str(next_x0.size())
        assert new_trajectory.size() == (traj_len - 2, self._space.n_x)

        self._trajectories_x0[traj_idx].copy_(new_x0)
        self._trajectories.append(Parameter(new_trajectory, requires_grad=True))
        self._trajectories_x0.append(Parameter(next_x0, requires_grad=True))

    def forward(self, traj_nums: Tensor, indices: Tensor) -> Tensor:
        """Returns a batch of trajectory states.

        Args:
            traj_nums: torch.int tensor ``(batch,)``
            indices: torch.int tensor ``(batch,)``
        """

        assert tensor_is_int(traj_nums) and tensor_is_int(
            indices
        ), "Inputs must be torch.int Tensors"
        assert (
            traj_nums.size() == indices.size()
        ), f"Batch dims don't match: ({traj_nums.size()}), ({indices.size()})"
        traj_nums_flat = traj_nums.flatten()
        indices_flat = indices.flatten()
        ret = torch.zeros(traj_nums.numel(), self._space.n_x)

        for idx, (traj_num, traj_index) in enumerate(zip(traj_nums_flat, indices_flat)):
            assert traj_num >= 0, f"Invalid trajectory number {traj_num}"
            assert traj_index >= 0, f"Invalid trajectory index {traj_index}"
            if traj_index == 0:
                ret[idx] = self._trajectories_x0[traj_num]
            elif traj_index == self._trajectories[traj_num].shape[0] + 1:
                ret[idx] = self._trajectories_x0[traj_num + 1]
            else:
                ret[idx] = self._trajectories[traj_num][traj_index - 1, :]

        return ret.reshape(traj_nums.size() + (self._space.n_x,))


### Unit tests
if __name__ == "__main__":
    test_space = FloatingBaseSpace(n_joints=0)
    traj = LearnableTrajectories(test_space)

    print("All Tests Passed!")
