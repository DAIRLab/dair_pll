"""Interface with Drake ``MultibodyPlant`` simulation.

Interfacing with Drake is done by massaging a drake system into the
``System`` interface defined in ``system.py`` with a new inheriting type,
``DrakeSystem``.

A large portion of the internal implementation of ``DrakeSystem`` is contained
in ``MultibodyPlantDiagram`` in ``drake_utils.py``.
"""
from typing import Callable, Tuple, Dict, List, Optional

import torch
from torch import Tensor
from tensordict.tensordict import TensorDict

from dair_pll.drake_state_converter import DrakeStateConverter
from dair_pll.drake_utils import MultibodyPlantDiagram
from dair_pll.integrator import StateIntegrator
from dair_pll.state_space import ProductSpace
from dair_pll.system import System
from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.plant import MultibodyPlant

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
                 visualization_file: Optional[str] = "meshcat",
                 additional_system_builders: List[Callable[[DiagramBuilder, MultibodyPlant], None]] = [],
                 g_frac: Optional[float] = 1.0) -> None:
        """Inits ``DrakeSystem`` with provided model URDFs.

        Args:
            urdfs: Names and corresponding URDFs to add as models to plant.
            dt: Time step of plant in seconds.
            visualization_file: Optional output GIF filename for trajectory
              visualization.
            additional_system_builders: Optional functions that add additional Drake
              Systems to the plant diagram.
        """
        plant_diagram = MultibodyPlantDiagram(urdfs, dt, visualization_file,
                                              additional_system_builders, g_frac=g_frac)

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

        carry_0 = self.populate_carry(carry_0)

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

    def populate_carry(self, carry: Tensor) -> Tensor:
        sim = self.plant_diagram.sim
        plant = self.plant_diagram.plant
        new_plant_context = plant.GetMyMutableContextFromRoot(
            sim.get_mutable_context())
        carry_next = torch.clone(carry.detach())
        if type(carry) == TensorDict:
            for key in carry.keys():
                if key == "contact_forces":
                    for body_name in carry[key].keys():
                        carry_next[key][body_name] = torch.zeros(3).reshape(1, -1)
                    contact_results = plant.get_contact_results_output_port().Eval(new_plant_context)
                    for idx in range(contact_results.num_point_pair_contacts()):
                        contact = contact_results.point_pair_contact_info(idx)
                        bodyA_name = plant.get_body(contact.bodyA_index()).name()
                        bodyB_name = plant.get_body(contact.bodyB_index()).name()
                        if bodyA_name in carry[key].keys():
                            carry_next[key][bodyA_name] -= Tensor(contact.contact_force()).reshape(1, -1)
                            print(f"Subtracting {contact.contact_force()} from {key}.{bodyA_name}")
                        elif bodyB_name in carry[key].keys():
                            carry_next[key][bodyB_name] += Tensor(contact.contact_force()).reshape(1, -1)
                            print(f"Adding {contact.contact_force()} to {key}.{bodyB_name}")
                        

                if plant.HasOutputPort(key):
                    carry_next[key] = plant.GetOutputPort(key).Eval(new_plant_context).reshape(1, -1)
                    print(f"Writing {carry_next[key]} to {key}")
        return carry_next

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

        carry_next = self.populate_carry(carry)
        input("Step...")

        return Tensor(x_next), carry_next
