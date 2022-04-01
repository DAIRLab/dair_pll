"""Interface with Drake ``MultibodyPlant`` simulation.

Interfacing with Drake is done by massaging a drake system into the
``System`` interface defined in ``system.py`` with a new inheriting type,
``DrakeSystem``.

A large portion of the internal implementation of ``DrakeSystem`` is contained
in ``MultibodyPlantDiagram`` in ``drake_utils.py``.
"""
from typing import Tuple, Dict

import torch
from torch import Tensor

from dair_pll.drake_state_converter import DrakeStateConverter
from dair_pll.drake_utils import MultibodyPlantDiagram
from dair_pll.integrator import StateIntegrator
from dair_pll.state_space import ProductSpace
from dair_pll.system import System


class DrakeSystem(System):
    """``System`` wrapper of a Drake simulation environment for a
    ``MultibodyPlant``.

    Drake simulation is constructed as a ``Simulator`` of a ``Diagram`` in a
    member ``MultibodyPlantDiagram`` variable. States are converted
    between ``StateSpace`` and Drake formats via ``DrakeStateConverter``.
    """
    plant_diagram: MultibodyPlantDiagram
    urdfs: Dict[str, str]
    dt: float
    space: ProductSpace

    def __init__(self,
                 urdfs: Dict[str, str],
                 dt: float,
                 enable_visualizer=False) -> None:
        """Inits ``DrakeSystem`` with provided model URDFs.

        Args:
            urdfs: Names and corresponding URDFs to add as models to plant.
            dt: Time step of plant in seconds.
            enable_visualizer: Whether to add visualization system to diagram.
        """
        plant_diagram = MultibodyPlantDiagram(urdfs, dt, enable_visualizer)

        space = plant_diagram.generate_state_space()
        integrator = StateIntegrator(space, self.sim_step, dt)

        super().__init__(space, integrator)
        self.plant_diagram = plant_diagram
        self.dt = dt
        self.urdfs = urdfs
        self.set_carry_sampler(lambda: Tensor([False]))

        # Drake simulations cannot be batched
        self.max_batch_dim = 0

    def preprocess_initial_condition(self, x_0: Tensor, carry_0: Tensor) -> \
            Tuple[Tensor, Tensor]:
        """Preprocesses initial condition state sequence into single state
        initial condition for integration.

        Args:
            x_0: (T_0, space.n_x) initial state sequence.
            carry_0: (1, ?) initial hidden state.

        Returns:
            (1, space.n_x) processed initial state.
            (1, ?) processed initial hidden state.
        """
        # select most recent state in this case and ensure tensor size
        # compatibility with call to ``System.preprocess_initial_condition``
        x_0, carry_0 = super().preprocess_initial_condition(x_0, carry_0)

        # Set state initial condition in internal Drake ``Simulator`` context.
        plant = self.plant_diagram.plant
        sim = self.plant_diagram.sim
        sim_context = sim.get_mutable_context()
        sim_context.SetTime(self.get_quantized_start_time(0.0))
        plant_context = plant.GetMyMutableContextFromRoot(
            sim.get_mutable_context())

        DrakeStateConverter.state_to_context(plant, plant_context,
                                             x_0.detach().numpy(),
                                             self.plant_diagram.model_ids,
                                             self.space)
        sim.Initialize()

        return x_0, carry_0

    def get_quantized_start_time(self, start_time: float) -> float:
        """Get phase-aligned start time for Drake ``Simulator``.

        As Drake models time stepping as events in a continuous time domain,
        some special care must be taken to ensure each call to
        ``DrakeSystem.step()`` triggers one update. This is done by
        offsetting the simulation duration to advance to ``N * dt + dt/4`` to
        prevent accidentally taking 2 or 0 steps with a call to ``step()``.

        Args:
            start_time: Time step beginning time.

        Returns:
            Time step quantized starting time.
        """
        dt = self.dt
        eps = dt / 4

        time_step_phase = start_time % dt
        offset = (dt if time_step_phase > (dt / 2.) else 0.) - time_step_phase
        cur_time_quantized = start_time + offset + eps

        return cur_time_quantized

    def sim_step(self, x: Tensor, carry: Tensor) -> Tuple[Tensor, Tensor]:
        """Simulate forward in time one step.

        Args:
            x: (n_x,) current state.
            carry: (?,) current hidden state.

        Returns:
            (n_x,) next state.
            (?,) next hidden state.
        """
        # pylint: disable=E1103
        assert x.shape == torch.Size([self.space.n_x])
        assert carry.dim() == 1

        sim = self.plant_diagram.sim
        plant = self.plant_diagram.plant

        # Advances one time step
        finishing_time = self.get_quantized_start_time(
            sim.get_mutable_context().get_time()) + self.dt
        sim.AdvanceTo(finishing_time)

        # Retrieves post-step state as numpy ndarray
        new_plant_context = plant.GetMyMutableContextFromRoot(
            sim.get_mutable_context())
        x_next = DrakeStateConverter.context_to_state(
            plant, new_plant_context, self.plant_diagram.model_ids, self.space)

        return Tensor(x_next), carry
