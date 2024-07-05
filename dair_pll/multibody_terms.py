"""Mathematical implementation of multibody dynamics terms calculations.

This file implements the :py:class:`MultibodyTerms` type, which interprets a
list of urdfs as a learnable Lagrangian system with contact, taking the state
space from the corresponding :py:class:`MultibodyPlantDiagram` as a given, and
interpreting the various inertial and geometric terms stored within it as
initial conditions of learnable parameters.

Multibody dynamics can be derived from four functions of state [q,v]:

    * M(q), the generalized mass-matrix
    * F(q), the non-contact/Lagrangian force terms.
    * phi(q), the signed distance between collision candidates.
    * J(q), the contact-frame velocity Jacobian between collision candidates.

The first two terms depend solely on state and inertial properties,
and parameterize the contact-free Lagrangian dynamics as::

    dv/dt = (M(q) ** (-1)) * F(q)

These terms are accordingly encapsulated in a :py:class:`LagrangianTerms`
instance.

The latter two terms depend solely on the geometry of bodies coming into
contact, and are encapsulated in a :py:class:`ContactTerms` instance.

For both sets of terms, we derive their functional form either directly or in
part through symbolic analysis of the :py:class:`MultibodyPlant` of the
associated :py:class:`MultibodyPlantDiagram`. The :py:class:`MultibodyTerms`
object manages the symbolic calculation and has corresponding
:py:class:`LagrangianTerms` and :py:class:`ContactTerms` members.
"""
from typing import List, Tuple, Callable, Dict, cast, Optional
from dataclasses import dataclass

import drake_pytorch  # type: ignore
import numpy as np
import torch
import pdb

from pydrake.geometry import SceneGraphInspector, GeometryId  # type: ignore
from pydrake.multibody.plant import MultibodyPlant_  # type: ignore
from pydrake.multibody.tree import JacobianWrtVariable  # type: ignore
from pydrake.multibody.tree import ModelInstanceIndex  # type: ignore
from pydrake.multibody.tree import SpatialInertia_, UnitInertia_, \
                                   RotationalInertia_  # type: ignore
from pydrake.symbolic import Expression, Variable  # type: ignore
from pydrake.symbolic import MakeVectorVariable, Jacobian  # type: ignore
from pydrake.systems.framework import Context  # type: ignore
from scipy.spatial.transform import Rotation
from torch import Tensor
from torch.nn import Module, ModuleList, Parameter, ParameterList

from dair_pll import drake_utils
from dair_pll.drake_utils import DrakeBody
from dair_pll.deep_support_function import extract_mesh_from_support_function, \
    get_mesh_summary_from_polygon
from dair_pll.drake_state_converter import DrakeStateConverter
from dair_pll.drake_utils import MultibodyPlantDiagram
from dair_pll.geometry import GeometryCollider, \
    PydrakeToCollisionGeometryFactory, \
    CollisionGeometry, DeepSupportConvex, Polygon, Box, \
    Plane, _NOMINAL_HALF_LENGTH
from dair_pll.inertia import InertialParameterConverter
from dair_pll.system import MeshSummary
from dair_pll.tensor_utils import (pbmm, deal, spatial_to_point_jacobian)

ConfigurationInertialCallback = Callable[[Tensor, Tensor], Tensor]
StateInputInertialCallback = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]

CENTER_OF_MASS_DOF = 3
INERTIA_TENSOR_DOF = 6
DEFAULT_SIMPLIFIER = drake_pytorch.Simplifier.QUICKTRIG

@dataclass
class InertiaLearn:
    """Class to specify which inertial parameters to learn"""
    mass: bool = False
    com: bool = False
    inertia: bool = False


# noinspection PyUnresolvedReferences
def init_symbolic_plant_context_and_state(
    plant_diagram: MultibodyPlantDiagram
) -> Tuple[MultibodyPlant_[Expression], Context, np.ndarray, np.ndarray]:
    """Generates a symbolic interface for a :py:class:`MultibodyPlantDiagram`.

    Generates a new Drake ``Expression`` data type state in
    :py:class:`StateSpace` format, and sets this state inside a new context for
    a symbolic version of the diagram's :py:class:`MultibodyPlant`.

    Args:
        plant_diagram: Drake MultibodyPlant diagram to convert to symbolic.

    Returns:
        New symbolic plant.
        New plant's context, with symbolic states set.
        (n_q,) symbolic :py:class:`StateSpace` configuration.
        (n_v,) symbolic :py:class:`StateSpace` velocity.
    """
    plant = plant_diagram.plant.ToSymbolic()
    space = plant_diagram.space
    context = plant.CreateDefaultContext()

    # :py:class:`StateSpace` representation of Plant's state.
    q = MakeVectorVariable(plant.num_positions(), 'q', Variable.Type.CONTINUOUS)
    v = MakeVectorVariable(plant.num_velocities(), 'v',
                           Variable.Type.CONTINUOUS)
    x = np.concatenate([q, v], axis=-1)

    # Set :py:class:`StateSpace` symbolic state inside
    DrakeStateConverter.state_to_context(plant, context, x,
                                         plant_diagram.model_ids, space)
    return plant, context, q, v


class LagrangianTerms(Module):
    """Container class for non-contact/Lagrangian dynamics terms.

    Accepts batched pytorch callback functions for M(q) and F(q) and related
    contact terms in ``theta`` format (see ``inertia.py``).
    """
    mass_matrix: Optional[ConfigurationInertialCallback]
    lagrangian_forces: Optional[StateInputInertialCallback]
    body_parameters: ParameterList
    inertial_parameters: Tensor

    def __init__(self, plant_diagram: MultibodyPlantDiagram,
                 inertia_learn: InertiaLearn = InertiaLearn(), constant_bodies: List[str] = []) -> None:
        """Inits :py:class:`LagrangianTerms` with prescribed parameters and
        functional forms.

        Args:
            plant_diagram: Drake MultibodyPlant diagram to extract terms from.
            inertia_mode: An integer 0, 1, 2, 3, or 4 representing the
            inertial parameters the model can learn.  The higher the number
            the more inertial parameters are free to be learned, and 0
            corresponds to learning no inertial parameters.
        """
        super().__init__()

        plant, context, q, v = init_symbolic_plant_context_and_state(
            plant_diagram)
        gamma = Jacobian(plant.GetVelocities(context), v)

        body_param_tensors, body_variables, bodies = \
            LagrangianTerms.extract_body_parameters_and_variables(
                plant, plant_diagram.model_ids, context)

        mass_matrix_expression = \
            gamma.T @ plant.CalcMassMatrixViaInverseDynamics(context) @ gamma

        self.mass_matrix, _ = drake_pytorch.sym_to_pytorch(
            mass_matrix_expression,
            q,
            body_variables,
            simplify_computation=DEFAULT_SIMPLIFIER)

        u = MakeVectorVariable(plant.num_actuated_dofs(), 'u',
                               Variable.Type.CONTINUOUS)
        drake_forces_expression = -plant.CalcBiasTerm(
            context) + plant.MakeActuationMatrix(
            ) @ u + plant.CalcGravityGeneralizedForces(context)

        lagrangian_forces_expression = gamma.T @ drake_forces_expression
        self.lagrangian_forces, _ = drake_pytorch.sym_to_pytorch(
            lagrangian_forces_expression,
            q,
            v,
            u,
            body_variables,
            simplify_computation=DEFAULT_SIMPLIFIER)

        # pylint: disable=E1103
        self.body_parameters = ParameterList()
        inertial_parameters = []
        for body_param_tensor, body in zip(body_param_tensors, bodies):
            learn_body = (body.name() not in constant_bodies)
            body_parameter = [
                Parameter(body_param_tensor[0], requires_grad=(learn_body and inertia_learn.mass)),
                Parameter(body_param_tensor[1:4], requires_grad=(learn_body and inertia_learn.com)),
                Parameter(body_param_tensor[4:], requires_grad=(learn_body and inertia_learn.inertia)),
            ]
            self.body_parameters.extend(body_parameter)
            inertial_parameters.append(torch.hstack(body_parameter))

        self.inertial_parameters = torch.stack(inertial_parameters)

    # noinspection PyUnresolvedReferences
    @staticmethod
    def extract_body_parameters_and_variables(
            plant: MultibodyPlant_[Expression],
            model_ids: List[ModelInstanceIndex],
            context: Context) -> Tuple[List[Tensor], np.ndarray, List[DrakeBody]]:
        """Generates parameterization and symbolic variables for all bodies.

        For a multibody plant, finds all bodies that should have inertial
        properties; extracts the current values as an initial condition for
        ``theta``-format learnable parameters, and sets new symbolic versions of
        these variables.

        Args:
            plant: Symbolic plant from which to extract parameterization.
            model_ids: List of models in plant.
            context: Plant's symbolic context.

        Returns:
            (n_bodies, 10) ``theta`` parameters initial conditions.
            (n_bodies, 10) symbolic inertial variables.
        """
        all_bodies, all_body_ids = drake_utils.get_all_inertial_bodies(
            plant, model_ids)

        body_parameter_list = []
        body_variable_list = []
        for body, body_id in zip(all_bodies, all_body_ids):
            mass = Variable(f'{body_id}_m', Variable.Type.CONTINUOUS)
            p_BoBcm_B = MakeVectorVariable(CENTER_OF_MASS_DOF, f'{body_id}_com',
                                           Variable.Type.CONTINUOUS)
            I_BBcm_B = MakeVectorVariable(INERTIA_TENSOR_DOF, f'{body_id}_I',
                                          Variable.Type.CONTINUOUS)

            # get original values
            body_parameter_list.append(
                InertialParameterConverter.drake_to_theta(
                    body.CalcSpatialInertiaInBodyFrame(context)))

            body_spatial_inertia = \
                SpatialInertia_[Expression].MakeFromCentralInertia(
                    mass=mass, p_PScm_E=p_BoBcm_B,
                    I_SScm_E=RotationalInertia_[Expression](*I_BBcm_B))

            body.SetMass(context, mass)
            body.SetSpatialInertiaInBodyFrame(context, body_spatial_inertia)
            body_variable_list.append(np.hstack((mass, p_BoBcm_B, I_BBcm_B)))
        # pylint: disable=E1103
        return body_parameter_list, np.vstack(body_variable_list), all_bodies

    def pi_cm(self) -> Tensor:
        """Returns inertial parameters in human-understandable ``pi_cm``
        -format"""
        return InertialParameterConverter.theta_to_pi_cm(self.inertial_parameters)

    def forward(self, q: Tensor, v: Tensor, u: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluates Lagrangian dynamics terms at given state and input.

        Args:
            q: (\*, n_q) configuration batch.
            v: (\*, n_v) velocity batch.
            u: (\*, n_u) input batch.

        Returns:
            (\*, n_v, n_v) mass matrix batch M(q)
            (\*, n_v) Lagrangian contact-free acceleration inv(M(q)) F(q)
        """
        # Pylint bug: cannot recognize instance attributes as Callable.
        # pylint: disable=not-callable
        assert self.mass_matrix is not None
        assert self.lagrangian_forces is not None
        inertia = \
            InertialParameterConverter.pi_cm_to_drake_spatial_inertia_vector(
            self.pi_cm())
        inertia = inertia.expand(q.shape[:-1] + inertia.shape)

        M = self.mass_matrix(q, inertia)
        non_contact_acceleration = torch.linalg.solve(
            M, self.lagrangian_forces(q, v, u, inertia))
        return M, non_contact_acceleration


ConfigurationCallback = Callable[[Tensor], Tensor]


def make_configuration_callback(expression: np.ndarray, q: np.ndarray) -> \
        Callable[[Tensor], Tensor]:
    """Converts drake symbolic expression to pytorch function via
    ``drake_pytorch``."""
    return cast(
        Callable[[Tensor], Tensor],
        drake_pytorch.sym_to_pytorch(
            expression, q, simplify_computation=DEFAULT_SIMPLIFIER)[0])


class ContactTerms(Module):
    """Container class for contact-related dynamics terms.

    Derives batched pytorch callback functions for collision geometry
    position and velocity kinematics from a
    :class:`~dair_pll.drake_utils.MultibodyPlantDiagram`.
    """
    geometry_rotations: Optional[ConfigurationCallback]
    geometry_translations: Optional[ConfigurationCallback]
    geometry_spatial_jacobians: Optional[ConfigurationCallback]
    geometries: ModuleList
    friction_param_list: ParameterList
    friction_params: Tensor
    collision_candidates: Tensor

    def __init__(self, plant_diagram: MultibodyPlantDiagram,
                 represent_geometry_as: str = 'box',
                 constant_bodies: List[str] = []) -> None:
        """Inits :py:class:`ContactTerms` with prescribed kinematics and
        geometries.

        phi(q) and J(q) are calculated implicitly from kinematics and ``n_g ==
        len(geometries)`` collision geometries C.

        Args:
            plant_diagram: Drake MultibodyPlant diagram to extract terms from.
            represent_geometry_as: How to represent the geometry of any
              learnable bodies (box/mesh/polygon).  By default, any ``Plane``
              objects are not considered learnable -- only boxes or meshes.
        """
        # pylint: disable=too-many-locals
        super().__init__()
        plant, context, q, v = init_symbolic_plant_context_and_state(
            plant_diagram)
        inspector = plant_diagram.scene_graph.model_inspector()

        collision_geometry_set = plant_diagram.collision_geometry_set
        geometry_ids = collision_geometry_set.ids
        coulomb_frictions = collision_geometry_set.frictions
        collision_candidates = collision_geometry_set.collision_candidates

        # sweep over collision elements
        geometries, rotations, translations, drake_spatial_jacobians = \
            ContactTerms.extract_geometries_and_kinematics(
                plant, inspector, geometry_ids, context, represent_geometry_as, constant_bodies=constant_bodies)

        for geometry_index, geometry_pair in enumerate(collision_candidates):
            if geometries[geometry_pair[0]] > geometries[geometry_pair[1]]:
                collision_candidates[geometry_index] = (geometry_pair[1],
                                                        geometry_pair[0])

        self.geometry_rotations = make_configuration_callback(
            np.stack(rotations), q)

        self.geometry_translations = make_configuration_callback(
            np.stack(translations), q)

        drake_velocity_jacobian = Jacobian(plant.GetVelocities(context), v)
        self.geometry_spatial_jacobians = make_configuration_callback(
            np.stack([
                jacobian @ drake_velocity_jacobian
                for jacobian in drake_spatial_jacobians
            ]), q)

        self.geometries = ModuleList(geometries)

        self.friction_param_list = ParameterList()
        for idx, friction in enumerate(coulomb_frictions):
            body = drake_utils.get_body_from_geometry_id(plant, inspector, geometry_ids[idx])
            learnable = (body.name() not in constant_bodies) and (body != plant.world_body())
            self.friction_param_list.append(Parameter(Tensor([friction.static_friction()]), requires_grad=learnable))
        self.friction_params = torch.hstack([param for param in self.friction_param_list])

        self.collision_candidates = Tensor(collision_candidates).t().long()

    def get_friction_coefficients(self) -> Tensor:
        """From the stored :py:attr:`friction_params`, compute the friction
        coefficient as its absolute value."""
        positive_friction_params = torch.abs(self.friction_params)

        return positive_friction_params

    # noinspection PyUnresolvedReferences
    @staticmethod
    def extract_geometries_and_kinematics(
        plant: MultibodyPlant_[Expression], inspector: SceneGraphInspector,
        geometry_ids: List[GeometryId], context: Context,
        represent_geometry_as: str, constant_bodies: List[str] = []
    ) -> Tuple[List[CollisionGeometry], List[np.ndarray], List[np.ndarray],
               List[np.ndarray]]:
        """Extracts modules and kinematics of list of geometries G.

        Args:
            plant: Multibody plant from which terms are extracted.
            inspector: Scene graph inspector associated with plant.
            geometry_ids: List of geometries to model.
            context: Plant's context with symbolic state.
            represent_geometry_as: How to represent learnable geometries.

        Returns:
            List of :py:class:`CollisionGeometry` models with one-to-one
              correspondence with provided geometries.
            List[(3,3)] of corresponding rotation matrices R_WG
            List[(3,)] of corresponding geometry frame origins p_WoGo_W
            List[(6,n_v)] of geometry spatial jacobians w.r.t. drake velocity
              coordinates, J(v_drake)_V_WG_W
        """
        world_frame = plant.world_frame()
        geometries = []
        rotations = []
        translations = []
        drake_spatial_jacobians = []

        for geometry_id in geometry_ids:
            geometry_pose = inspector.GetPoseInFrame(
                geometry_id).cast[Expression]()

            body = plant.GetBodyFromFrameId(
                inspector.GetFrameId(geometry_id))

            learnable = (body.name() not in constant_bodies) and (body != plant.world_body())

            geometry_frame = body.body_frame()

            geometry_transform = geometry_frame.CalcPoseInWorld(
                context) @ geometry_pose

            rotations.append(geometry_transform.rotation().matrix())

            translations.append(geometry_transform.translation())

            drake_spatial_jacobian = plant.CalcJacobianSpatialVelocity(
                context=context,
                with_respect_to=JacobianWrtVariable.kV,
                frame_B=geometry_frame,
                p_BoBp_B=geometry_pose.translation().reshape(3, 1),
                frame_A=world_frame,
                frame_E=world_frame)
            drake_spatial_jacobians.append(drake_spatial_jacobian)

            geometries.append(
                PydrakeToCollisionGeometryFactory.convert(
                    inspector.GetShape(geometry_id), represent_geometry_as,
                    learnable, body.name()))

        return geometries, rotations, translations, drake_spatial_jacobians

    @staticmethod
    def assemble_velocity_jacobian(R_CW, Jv_V_WC_W, p_CoCc_C):
        """Helper method to generate velocity jacobian from contact information.

        Args:
            R_CW: (\*, n_c, 3, 3) Rotation of world frame w.r.t. geometry frame.
            Jv_V_WC_W: (\*, 1, 6, n_v) Geometry spatial velocity Jacobian.
            p_CoCc_C: (\*, n_c, 3) Geometry-frame contact points.

        Returns:
            (\*, n_c, 3, n_v) World-frame contact point translational velocity
            Jacobian.
        """
        p_CoCc_W = pbmm(p_CoCc_C.unsqueeze(-2), R_CW).squeeze(-2)
        Jv_v_WCc_W = pbmm(spatial_to_point_jacobian(p_CoCc_W), Jv_V_WC_W)
        return Jv_v_WCc_W

    @staticmethod
    def relative_velocity_to_contact_jacobian(Jv_v_W_BcAc_F: Tensor,
                                              mu: Tensor) -> Tensor:
        """Helper method to reorder contact Jacobian columns.

        Args:
            Jv_v_W_BcAc_F: (\*, n_collisions, 3, n_v) collection of
            contact-frame relative velocity Jacobians.
            mu: (n_collisions,) list of

        Returns:
            (\*, 3 * n_collisions, n_v) contact jacobian J(q) in [J_n; mu * J_t]
            ordering.
        """
        # Tuple of (*, n_collisions, n_v)
        J_x, J_y, J_z = deal(Jv_v_W_BcAc_F, -2)

        J_n = J_z

        # Reshape (*, n_collisions, 2 * n_v) -> (*, 2 * n_collisions, n_v)
        # pylint: disable=E1103
        mu_shape = torch.Size((1,) * (J_x.dim() - 2) + mu.shape + (1,))
        friction_jacobian_shape = J_x.shape[:-2] + (-1, J_x.shape[-1])
        J_t = (mu.reshape(mu_shape) * torch.cat((J_x, J_y), dim=-1)) \
            .reshape(friction_jacobian_shape)
        return torch.cat((J_n, J_t), dim=-2)

    def forward(self, q: Tensor) -> Tuple[Tensor, Tensor]:
        """Evaluates Lagrangian dynamics terms at given state and input.

        Uses :py:class:`GeometryCollider` and kinematics to construct signed
        distance phi(q) and the corresponding Jacobian J(q).

        phi(q) and J(q) are calculated implicitly from kinematics and collision
        geometries.

        Args:
            q: (\*, n_q) configuration batch.
            indices that can collide.

        Returns:
            (\*, n_collisions) signed distance phi(q).
            (\*, 3 * n_collisions, n_v) contact Jacobian J(q).
        """
        # Pylint bug: cannot recognize instance attributes as Callable.
        # pylint: disable=too-many-locals,not-callable
        assert self.geometry_rotations is not None
        assert self.geometry_translations is not None
        assert self.geometry_spatial_jacobians is not None
        R_WC = self.geometry_rotations(q)
        p_WoCo_W = self.geometry_translations(q)
        Jv_V_WC_W = self.geometry_spatial_jacobians(q)

        indices_a = self.collision_candidates[0, :]
        indices_b = self.collision_candidates[1, :]

        geometries_a = [
            cast(CollisionGeometry, self.geometries[element_index])
            for element_index in indices_a
        ]
        geometries_b = [
            cast(CollisionGeometry, self.geometries[element_index])
            for element_index in indices_b
        ]

        friction_coefficients = self.get_friction_coefficients()
        mu_a = friction_coefficients[indices_a]
        mu_b = friction_coefficients[indices_b]

        # combine friction coefficients as in Drake.
        mu = (2 * mu_a * mu_b) / (mu_a + mu_b)

        R_WA = R_WC[..., indices_a, :, :]
        R_AW = deal(R_WA.transpose(-1, -2), -3)
        R_BW = deal(R_WC[..., indices_b, :, :].transpose(-1, -2), -3)

        Jv_V_WA_W = deal(Jv_V_WC_W[..., indices_a, :, :], -3, keep_dim=True)
        Jv_V_WB_W = deal(Jv_V_WC_W[..., indices_b, :, :], -3, keep_dim=True)

        # Interbody translation in A frame, shape (*, n_g, 3)
        p_AoBo_W = p_WoCo_W[..., indices_b, :] - p_WoCo_W[..., indices_a, :]
        p_AoBo_A = deal(pbmm(p_AoBo_W.unsqueeze(-2), R_WA).squeeze(-2), -2)

        Jv_v_W_BcAc_F = []
        phi_list = []
        obj_pair_list = []
        R_FW_list = []

        # bundle all modules and kinematics into a tuple iterator
        a_b = zip(geometries_a, geometries_b, R_AW, R_BW, p_AoBo_A, Jv_V_WA_W,
                  Jv_V_WB_W)

        # iterate over body pairs (Ai, Bi)
        for geo_a, geo_b, R_AiW, R_BiW, p_AiBi_A, Jv_V_WAi_W, Jv_V_WBi_W in a_b:
            # relative rotation between Ai and Bi, (*, 3, 3)
            R_AiBi = pbmm(R_AiW, R_BiW.transpose(-1, -2))

            # collision result,
            # Tuple[(*, n_c), (*, n_c, 3, 3), (*, n_c, 3), (*, n_c, 3)]
            phi_i, R_AiF, p_AiAc_A, p_BiBc_B = GeometryCollider.collide(
                geo_a, geo_b, R_AiBi, p_AiBi_A)
            n_c = phi_i.shape[1]

            # contact frame rotation, (*, n_c, 3, 3)
            R_FW = pbmm(R_AiF.transpose(-1, -2), R_AiW.unsqueeze(-3))

            # contact point velocity jacobians, (*, n_c, 3, n_v)
            Jv_v_WAc_W = ContactTerms.assemble_velocity_jacobian(
                R_AiW.unsqueeze(-3), Jv_V_WAi_W, p_AiAc_A)
            Jv_v_WBc_W = ContactTerms.assemble_velocity_jacobian(
                R_BiW.unsqueeze(-3), Jv_V_WBi_W, p_BiBc_B)

            # contact relative velocity, (*, n_c, 3, 3)
            Jv_v_W_BcAc_F.append(pbmm(R_FW, Jv_v_WBc_W - Jv_v_WAc_W))
            phi_list.append(phi_i)
            obj_pair_list.extend(n_c * [(geo_a.name, geo_b.name)])
            R_FW_list.extend([R_FW[..., i, :, :] for i in range(n_c)])

        # pylint: disable=E1103
        mu_repeated = torch.cat(
            [mu_i.repeat(phi_i.shape[-1]) for phi_i, mu_i in zip(phi_list, mu)])
        phi = torch.cat(phi_list, dim=-1)  # type: Tensor
        J = ContactTerms.relative_velocity_to_contact_jacobian(
            torch.cat(Jv_v_W_BcAc_F, dim=-3), mu_repeated)

        return phi, J, obj_pair_list, R_FW_list


class MultibodyTerms(Module):
    """Derives and manages computation of terms of multibody dynamics with
    contact.

    Primarily
    """
    lagrangian_terms: LagrangianTerms
    contact_terms: ContactTerms
    geometry_body_assignment: Dict[str, List[int]]
    plant_diagram: MultibodyPlantDiagram
    urdfs: Dict[str, str]
    inertia_mode: InertiaLearn

    def scalars_and_meshes(
            self) -> Tuple[Dict[str, float], Dict[str, MeshSummary]]:
        """Generates summary statistics for inertial and geometric quantities."""
        scalars = {}
        meshes = {}
        _, all_body_ids = \
            drake_utils.get_all_inertial_bodies(
                self.plant_diagram.plant,
                self.plant_diagram.model_ids)

        friction_coefficients = self.contact_terms.get_friction_coefficients()

        for body_pi, body_id in zip(self.lagrangian_terms.pi_cm(), all_body_ids):
            body_scalars = InertialParameterConverter.pi_cm_to_scalars(body_pi)

            scalars.update({
                f'{body_id}_{scalar_name}': scalar
                for scalar_name, scalar in body_scalars.items()
            })

            for geometry_index in self.geometry_body_assignment[body_id]:
                # include geometry
                geometry = self.contact_terms.geometries[geometry_index]
                geometry_scalars = geometry.scalars()
                scalars.update({
                    f'{body_id}_{scalar_name}': scalar
                    for scalar_name, scalar in geometry_scalars.items()
                })

                # include friction
                scalars[f'{body_id}_mu'] = \
                    friction_coefficients[geometry_index].item()

                geometry_mesh = None
                if isinstance(geometry, DeepSupportConvex):
                    geometry_mesh = extract_mesh_from_support_function(
                        geometry.network)

                elif isinstance(geometry, Polygon):
                    geometry_mesh = get_mesh_summary_from_polygon(geometry)

                if geometry_mesh != None:
                    meshes[body_id] = geometry_mesh
                    vertices = geometry_mesh.vertices
                    diameters = vertices.max(dim=0).values - vertices.min(
                        dim=0).values
                    center = vertices.min(dim=0).values + diameters / 2
                    scalars.update({
                        f'{body_id}_diameter_{axis}': value.item()
                        for axis, value in zip(['x', 'y', 'z'], diameters)
                    })
                    scalars.update({
                        f'{body_id}_center_{axis}': value.item()
                        for axis, value in zip(['x', 'y', 'z'], center)
                    })

        return scalars, meshes

    def forward(self, q: Tensor, v: Tensor,
                u: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Evaluates multibody system dynamics terms at given state and input.

        Calculation is performed as a thin wrapper around
        :py:class:`LagrangianTerms` and :py:class:`ContactTerms`. For
        convenience, this function also returns the Delassus operator
        `D(q) = J(q)^T inv(M(q)) J(q)`.

        Args:
            q: (\*, n_q) configuration batch.
            v: (\*, n_v) velocity batch.
            u: (\*, n_u) input batch.

        Returns:
            (\*, 3 * n_collisions, 3 * n_collisions) Delassus operator D(q).
            (\*, n_v, n_v) mass matrix batch M(q).
            (\*, 3 * n_collisions, n_v) contact Jacobian J(q).
            (\*, n_collisions) signed distance phi(q).
            (\*, n_v) Contact-free acceleration inv(M(q)) * F(q).
        """
        M, non_contact_acceleration = self.lagrangian_terms(q, v, u)
        phi, J, obj_pair_list, R_FW_list = self.contact_terms(q)

        delassus = pbmm(J, torch.linalg.solve(M, J.transpose(-1, -2)))
        return delassus, M, J, phi, non_contact_acceleration, obj_pair_list, R_FW_list

    def __init__(self, urdfs: Dict[str, str], inertia_mode: InertiaLearn = InertiaLearn(),
                 constant_bodies: List[str] = [],
                 represent_geometry_as: str = 'box',
                 randomize_initialization: bool = False,
                 g_frac: float = 1.0) -> None:
        """Inits :py:class:`MultibodyTerms` for system described in URDFs

        Interpretation is performed as a thin wrapper around
        :py:class:`LagrangianTerms` and :py:class:`ContactTerms`.

        As this module is also responsible for evaluating updated URDF
        representations, the associations between bodies and geometries is
        also tracked to enable URDF rendering in
        ``MultibodyTerms.EvalUrdfRepresentation`` and Tensorboard logging in
        ``MultibodyTerms.scalars``.

        Args:
            urdfs: Dictionary of named URDF XML file names, containing
              description of multibody system.
            inertia_mode: specify which inertial parameters to leran
            constant_bodies: list of body names to keep constant / not learn
            represent_geometry_as: String box/mesh/polygon to determine how
              the geometry should be represented.
        """
        super().__init__()

        plant_diagram = MultibodyPlantDiagram(urdfs, g_frac=g_frac)
        plant = plant_diagram.plant.ToSymbolic()
        inspector = plant_diagram.scene_graph.model_inspector()

        _, all_body_ids = drake_utils.get_all_bodies(plant,
                                                     plant_diagram.model_ids)

        # sweep over collision elements
        geometry_body_assignment: Dict[str, List[int]] = {
            body_id: [] for body_id in all_body_ids
        }

        geometry_ids = plant_diagram.collision_geometry_set.ids

        for geometry_index, geometry_id in enumerate(geometry_ids):
            geometry_frame_id = inspector.GetFrameId(geometry_id)
            geometry_body = plant.GetBodyFromFrameId(geometry_frame_id)
            geometry_body_identifier = drake_utils.unique_body_identifier(
                plant, geometry_body)
            geometry_body_assignment[geometry_body_identifier].append(
                geometry_index)

        # setup parameterization
        self.lagrangian_terms = LagrangianTerms(plant_diagram, inertia_mode, constant_bodies)
        self.contact_terms = ContactTerms(plant_diagram, represent_geometry_as, constant_bodies)
        self.geometry_body_assignment = geometry_body_assignment
        self.plant_diagram = plant_diagram
        self.urdfs = urdfs

    def randomize_multibody_terms(self, inertia_int) -> None:
        r"""Adds random noise to multibody terms in the following ways:
            - Geometry lengths can be between 0.5 and 1.5 times their original
              length (except DeepSupportConvex types, which always start as a
              small sphere).
            - Friction can be between 0.1 and 1.9 times their original size.
            - Total mass does not change.
            - Inertia is determined via:
                - Choose a random set of three length scales between 0.5 and 1.5
                  times the true length.
                - Choose a random mass fraction :math:`\nu` between 0.5 and 1.5.
                - Define :math:`I_{xx,princ.axis}` along a principal axis as
                  :math:`\frac{\nu m}{12} (l_y^2 + l_z^2)`, and similarly for
                  :math:`I_{yy,princ.axis}` and :math:`I_{zz,princ.axis}`.
                - Generate a random rotation in SO(3) and transform the
                  principal axis inertia matrix with it.
                - Define all moments and products of inertia from this rotated
                  version.
        """
        def scale_factory(x: Tensor) -> Tensor:
            """Return a scaled version of the input such that each element in
            the input individually is randomly scaled between 50% and 150% of
            its original value."""
            return torch.mul(x, torch.rand_like(x) + 0.5)

        # First do friction all at once.  Note that 
        # self.contact_terms.get_friction_coefficients will ensure the ground's
        # friction gets properly rewritten to 1.0.
        new_friction_params = scale_factory(self.contact_terms.friction_params)
        new_friction_params[1] = 1.0   # hack, ground is always element 1
        self.contact_terms.friction_params = Parameter(
            new_friction_params, requires_grad=True)

        # Second, randomize the geometry.
        for geometry in self.contact_terms.geometries:
            # Don't make changes for ground geometry.
            if isinstance(geometry, Plane):
                continue
            elif isinstance(geometry, Box):
                geometry.length_params = \
                    Parameter(scale_factory(geometry.length_params),
                              requires_grad=True)
            elif isinstance(geometry, Polygon):
                geometry.vertices_parameter = \
                    Parameter(scale_factory(geometry.vertices_parameter),
                              requires_grad=True)
            elif isinstance(geometry, DeepSupportConvex):
                # Deep support convex cannot be randomized.  However, it always
                # starts as a small ball and thus never matches the true
                # geometry anyway, so it's fine to skip random initialization
                # for this type.
                pass
            else:
                raise NotImplementedError("Can only randomize Box and Polygon "
                    "geometries; intentionally does nothing for "
                    "DeepSupportConvex; can't handle other geometry types.")

        # Third, randomize the inertia.  Only randomize the learnable params.
        if INERTIA_PARAM_OPTIONS[inertia_int] != 'none' and \
           INERTIA_PARAM_OPTIONS[inertia_int] != 'masses':
            # Use a while loop to prevent nan situations.
            keep_trying = True
            while keep_trying:
                n_bodies = self.lagrangian_terms.inertial_parameters.shape[0]
                for idx in range(n_bodies):
                    pi_cm = self.lagrangian_terms.original_pi_cm_params[idx]

                    # Let the center of mass be anywhere within the inner half
                    # of a nominal geometry.
                    mass = pi_cm[0].item()
                    pi_cm[1:4] += mass*(torch.rand(3) - 0.5) \
                        * _NOMINAL_HALF_LENGTH

                    if INERTIA_PARAM_OPTIONS[inertia_int] == 'all':
                        # Define the moments of inertia assuming a solid block
                        # of homogeneous density with random mass and random
                        # lengths.
                        rand_mass = mass * (torch.rand(1) + 0.5)
                        rand_lens = (torch.rand(3) + 0.5) * _NOMINAL_HALF_LENGTH
                        scaling = rand_mass/12
                        Ixx_pa = scaling * (rand_lens[1]**2 + rand_lens[2]**2)
                        Iyy_pa = scaling * (rand_lens[0]**2 + rand_lens[2]**2)
                        Izz_pa = scaling * (rand_lens[0]**2 + rand_lens[1]**2)

                        # Randomly rotate the principal axes.
                        rot_mat = Tensor(Rotation.random().as_matrix())
                        I_mat_pa = Tensor([[Ixx_pa, 0., 0.],
                                           [0., Iyy_pa, 0.],
                                           [0., 0., Izz_pa]])
                        I_rand = rot_mat.T @ I_mat_pa @ rot_mat

                        # Grab the moments and products of inertia from this
                        # result.
                        Ixx, Iyy, Izz = I_rand[0,0], I_rand[1,1], I_rand[2,2]
                        Ixy, Ixz, Iyz = I_rand[0,1], I_rand[1,2], I_rand[1,2]

                        pi_cm[4:7] = Tensor([Ixx, Iyy, Izz])
                        pi_cm[7:10] = Tensor([Ixy, Ixz, Iyz])

                    self.lagrangian_terms.original_pi_cm_params[idx] = pi_cm

                new_theta_params = InertialParameterConverter.pi_cm_to_theta(
                    self.lagrangian_terms.original_pi_cm_params)
                if not torch.any(torch.isnan(new_theta_params)):
                    keep_trying = False

        new_theta_params = InertialParameterConverter.pi_cm_to_theta(
            self.lagrangian_terms.original_pi_cm_params)

        self.lagrangian_terms.inertial_parameters = Parameter(
            new_theta_params, requires_grad=True)
