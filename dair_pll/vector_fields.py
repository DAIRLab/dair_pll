"""Definitions for randomly generated force vector fields for testing the
capabilities of residual physics in simulation.
"""

from abc import ABC, abstractmethod
import pdb
from typing import Tuple

import numpy as np

from pydrake.systems.framework import LeafSystem



ROTATION_PRIMITIVE = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
INWARD_PRIMITIVE = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 0]])


class ForceVectorField(ABC):
    """Class that keeps track of artificially created force vector fields."""

    def __init__(self, n_velocity: int) -> None:
        self.n_velocity = n_velocity

    @abstractmethod
    def generalized_force_by_state(self, state: np.ndarray) -> np.ndarray:
        """Given a system state, return a generalized force."""
        pass


class VortexForceVectorField(ForceVectorField):
    """Specifically generate a vortex like a toilet bowl."""

    def __init__(self, n_velocity: int,
                 center_xy: Tuple[float, float] = (0., 0.),
                 rotation_scaling: float = 1.,
                 inward_scaling: float = 1.,
                 height_std_dev: float = 1.) -> None:
        super().__init__(n_velocity)

        self.center_x = center_xy[0]
        self.center_y = center_xy[1]
        self.w_rot = rotation_scaling
        self.w_in = inward_scaling
        self.z_std_dev = height_std_dev

    def generalized_force_by_state(self, state: np.ndarray) -> np.ndarray:
        """Given a system state, return a generalized force.  For this vortex
        force vector field, the generalized force depends only on the system's
        location in space.

        TODO:
         - handle batched configurations
         - factor z height into it
         - maybe add a z torque as well:  be careful, this would need to be in
            world z coordinates, not body (just need to check if this uses
            world or body coordinates, not sure).
        """
        xyz_loc = state[4:7]

        xy_mag = np.linalg.norm(xyz_loc[:2]) + 1e-4
        rotation_mat = self.w_rot * ROTATION_PRIMITIVE / xy_mag
        inward_mat = self.w_in * INWARD_PRIMITIVE / xy_mag

        force = (rotation_mat + inward_mat) @ xyz_loc

        generalized_force = np.zeros((self.n_velocity))
        generalized_force[3:6] = force

        return generalized_force


class ViscousDampingVectorField(ForceVectorField):
    """Specifically add viscous damping to linear, angular, and articulation
    velocities."""

    def __init__(self, n_velocity: int, w_linear: float = 0.0,
                 w_angular: float = 0.0, w_articulation: float = 0.0):
        super().__init__(n_velocity)

        self.w_linear = w_linear
        self.w_angular = w_angular
        self.w_articulation = w_articulation

    def generalized_force_by_state(self, state: np.ndarray) -> np.ndarray:
        """Given a system state, return a generalized force.  For this viscous
        damping vector field, the generalized forces depend only on the
        velocity components of the state.
        """
        vels = state[-self.n_velocity:]

        if np.any(np.isnan(vels)):
            pdb.set_trace()

        generalized_force = np.concatenate((
            -self.w_linear * vels[:3],
            -self.w_angular * vels[3:6],
            -self.w_articulation * vels[6:]))

        return generalized_force



class ForceVectorFieldInjectorLeafSystem(LeafSystem):
    """Create a Drake ``LeafSystem`` which can inject forces from a force vector
    field into the dynamics of a Multibody Plant.
    """
    def __init__(self, n_state: int, n_velocity: int,
                 vector_field: ForceVectorField):
        super().__init__()

        # Store the force vector field.
        self.vector_field = vector_field

        # Create an input port for the current state of the system.
        self.mbp_state_input_port = self.DeclareVectorInputPort(
            name="mbp_state", size=n_state)

        # Create an output port for the generalized forces.
        self.DeclareVectorOutputPort(name="force_vector", size=n_velocity,
                                     calc=self.CalculateVectorField)

    def CalculateVectorField(self, context, output):
        # Evaluate the input ports to obtain the current multibody plant state.
        mbp_state = self.mbp_state_input_port.Eval(context)

        # Generate the generalized force from the multibody plant state.
        generalized_force = \
            self.vector_field.generalized_force_by_state(mbp_state)

        # Write the output vector.
        output.SetFromVector(generalized_force)

