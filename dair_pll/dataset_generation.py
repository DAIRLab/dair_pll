r""" This module also contains utilities for generating simulation data
from a :class:`~dair_pll.system.System`\ .

Centers around the :class:`ExperimentDatasetGenerator` type, which takes in a
configuration describing trajectory parameters, as well as distributions for
initial conditions and noise to apply to simulated states.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict, Type, Union, List

import pdb

import torch
from torch import Tensor
from tensordict.tensordict import TensorDict

from dair_pll import file_utils
from dair_pll.state_space import (
    ConstantSampler,
    CenteredSampler,
    UniformSampler,
    GaussianWhiteNoiser,
    UniformWhiteNoiser,
    StateSpaceSampler,
)
from dair_pll.system import System

DEFAULT_TRAJECTORY_BATCH_SIZE = 30
"""Number of trajectories to simulate before intermediate save-to-disk."""


@dataclass
class DataGenerationConfig:
    """:func:`~dataclasses.dataclass` for configuring generation of a
    trajectory dataset."""

    # pylint: disable=too-many-instance-attributes
    dt: float = 1e-3
    r"""Time step, ``> 0``\ ."""
    n_pop: float = 16384
    r"""Total number of trajectories to select from, ``>= 0``\ ."""
    trajectory_length: int = 80
    r"""Trajectory length, ``>= 1``\ ."""
    x_0: Tensor = field(default_factory=lambda: torch.tensor([]))
    """Nominal initial states."""
    sampler_type: Type[StateSpaceSampler] = ConstantSampler
    r"""Distribution for sampling around :attr:`x_0`\ ."""
    sampler_kwargs: Dict[str, Any] = field(default_factory=dict)
    r"""``(2 * n_v)`` size of perturbations sampled around :attr:`x_0`\ ."""
    noiser_type: Optional[
        Union[Type[GaussianWhiteNoiser], Type[UniformWhiteNoiser]]
    ] = GaussianWhiteNoiser
    """Type of noise to add to data."""
    static_noise: Tensor = field(default_factory=lambda: torch.tensor([]))
    """``(2 * n_v)`` sampler ranges for constant-in-time trajectory noise."""
    dynamic_noise: Tensor = field(default_factory=lambda: torch.tensor([]))
    """``(2 * n_v)`` sampler ranges for i.i.d.-in-time trajectory noise."""
    storage: str = "./"
    """Experiment folder for data storage. Defaults to working directory."""

    def __post_init__(self):
        """Method to check validity of parameters."""
        assert self.dt > 0.0
        assert self.n_pop >= 0
        assert self.trajectory_length >= 1
        if self.noiser_type:
            assert self.x_0.nelement() == self.static_noise.nelement()
            assert self.x_0.nelement() == self.dynamic_noise.nelement()


class ExperimentDatasetGenerator:
    r"""Conducts generation of a population of simulated trajectories from
    a provided system.

    These trajectories are stored on disk at a location described in the
    generator's configuration. Two sets are generated: one `ground truth`
    set, which are precisely what the system predicts; and a `learning` set
    which has artificial measurement noise added.

    The various parameters describing the qualities of the dataset are given
    in the provided :py:class:`DataGenerationConfig`\ .
    """

    system: System
    config: DataGenerationConfig

    def __init__(self, system: System, config: DataGenerationConfig) -> None:
        self.system = system
        self.config = config

    def generate(self) -> None:
        """Simulate trajectories and write them to disk."""
        config = self.config
        n_pop = config.n_pop
        ground_truth_dir = file_utils.ground_truth_data_dir(config.storage)
        learning_dir = file_utils.learning_data_dir(config.storage)
        n_on_disk = file_utils.get_trajectory_count(ground_truth_dir)
        n_to_add = DEFAULT_TRAJECTORY_BATCH_SIZE
        while n_on_disk < n_pop:
            n_on_disk = file_utils.get_trajectory_count(ground_truth_dir)
            n_to_add = min(n_to_add, max(n_pop - n_on_disk, 0))
            if n_to_add == 0:
                break
            ground_truth_trajectories = self.simulate_trajectory_set(n_to_add)
            learning_trajectories = self.make_noised_trajectories(
                ground_truth_trajectories
            )
            for relative_index in range(n_to_add):
                ground_truth_file = file_utils.trajectory_file(
                    ground_truth_dir, n_on_disk + relative_index
                )
                torch.save(ground_truth_trajectories[relative_index], ground_truth_file)

                learning_file = file_utils.trajectory_file(
                    learning_dir, n_on_disk + relative_index
                )
                torch.save(learning_trajectories[relative_index], learning_file)

    def simulate_trajectory_set(self, num_trajectories: int) -> List[Tensor]:
        """Simulate trajectories using :py:attr:`system`

        Args:
            num_trajectories: number of trajectories to simulate

        Returns:
            List of ``(T, self.system.space.n_x)`` trajectories.
        """
        assert num_trajectories >= 0
        config = self.config
        system = self.system
        starting_state = config.x_0
        system.set_state_sampler(
            config.sampler_type(system.space, **config.sampler_kwargs)
        )

        trajectories = []
        for _ in range(num_trajectories):
            trajectory, carry_traj = system.sample_trajectory(config.trajectory_length)
            if type(carry_traj) == TensorDict:
                # Save as Tensordict instead of Trajetory
                carry_traj["state"] = trajectory.reshape(
                    config.trajectory_length + 1, 1, -1
                )
                trajectory = carry_traj
            trajectories.append(trajectory)
        return trajectories

    def make_noised_trajectories(self, traj_set: List[Tensor]) -> List[Tensor]:
        r"""Given ground-truth trajectories predicted with :py:attr:`system`\ ,
        returns corresponding set of learning trajectories with added
        measurement noise.

        Args:
            traj_set: List of ground-truth ``(*, self.system.space.n_x)`` state
              trajectories.

        Returns:
            List of ``(*, self.system.space.n_x)`` noisy state trajectories.
        """
        config = self.config
        if not config.noiser_type:
            return traj_set
        noiser = config.noiser_type(self.system.space)
        noised_trajectories = []
        for traj in traj_set:
            static_disturbed = noiser.noise(
                traj, config.static_noise, independent=False
            )
            dynamic_disturbed = noiser.noise(static_disturbed, config.dynamic_noise)
            dynamic_disturbed = self.system.space.project_derivative(
                dynamic_disturbed, config.dt
            )
            noised_trajectories.append(dynamic_disturbed)
        return noised_trajectories
