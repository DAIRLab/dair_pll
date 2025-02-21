"""Drake simulation setup for multibody systems.

This file implements :py:class:`MultibodyPlantDiagram`, which instantiates
Drake simulation and visualization system for a given group of URDF files.

Visualization is done via Drake's VideoWriter. Details on using the VideoWriter
are available in the documentation for :py:mod:`dair_pll.vis_utils`.

In order to make the Drake states compatible with available
:py:class:`~dair_pll.state_space.StateSpace` inheriting classes,
users must define the drake system by a collection of URDF files, each of
which contains a model for exactly one floating- or fixed-base rigid
multibody chain. This allows for the system to be modeled as having a
:py:class:`~dair_pll.state_space.ProductSpace` state space, where each
factor space is a
:py:class:`~dair_pll.state_space.FloatingBaseSpace`
or :py:class:`~dair_pll.state_space.FixedBaseSpace`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Tuple, Dict, List, Optional, Union, Type, cast, TypeAlias

import math
import enum
import gin

import numpy as np

# TODO: put in place
from pydrake.all import MeshcatVisualizerParams, StartMeshcat, BodyIndex, ContactResults, Value  # type: ignore

from pydrake.autodiffutils import AutoDiffXd  # type: ignore
from pydrake.geometry import HalfSpace, SceneGraph  # type: ignore

# pylint: disable-next=import-error
from pydrake.geometry import SceneGraphInspector_, GeometryId  # type: ignore
from pydrake.math import RigidTransform, RollPitchYaw, RigidTransform_  # type: ignore
from pydrake.multibody.parsing import Parser  # type: ignore
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    CoulombFriction_,
    ContactModel,
    DiscreteContactApproximation,
)  # type: ignore
from pydrake.multibody.plant import CoulombFriction  # type: ignore
from pydrake.multibody.plant import MultibodyPlant  # type: ignore
from pydrake.multibody.plant import MultibodyPlant_  # type: ignore
from pydrake.multibody.tree import ModelInstanceIndex  # type: ignore
from pydrake.multibody.tree import SpatialInertia_  # type: ignore
from pydrake.multibody.tree import world_model_instance, Body_  # type: ignore
from pydrake.symbolic import Expression  # type: ignore
from pydrake.systems.analysis import Simulator  # type: ignore
from pydrake.systems.framework import DiagramBuilder, DiagramBuilder_, LeafSystem  # type: ignore
from pydrake.geometry import MeshcatVisualizer, Role

# pylint: disable-next=import-error
from pydrake.visualization import VideoWriter  # type: ignore

from dair_pll import state_space

WORLD_GROUND_PLANE_NAME = "world_ground_plane"
DRAKE_MATERIAL_GROUP = "material"
DRAKE_FRICTION_PROPERTY = "coulomb_friction"
N_DRAKE_FLOATING_BODY_VELOCITIES = 6

GROUND_COLOR = np.array([0.5, 0.5, 0.5, 0.1])

CAM_FOV = np.pi / 6
VIDEO_PIXELS = [960, 1280]
FPS = 30

# dt of underlying sim
SIM_DT = 1e-4

# TODO currently hard-coded camera pose could eventually be dynamically chosen
# to fit the actual trajectory.
SENSOR_RPY = np.array([-np.pi / 2, 0.0, 0.0])
SENSOR_POSITION = np.array([0.0, -1.0, 0.2])
SENSOR_POSE = RigidTransform(RollPitchYaw(SENSOR_RPY).ToQuaternion(), SENSOR_POSITION)

MultibodyPlantFloat: TypeAlias = cast(Type, MultibodyPlant_[float])
MultibodyPlantAutoDiffXd: TypeAlias = cast(Type, MultibodyPlant_[AutoDiffXd])
MultibodyPlantExpression: TypeAlias = cast(Type, MultibodyPlant_[Expression])
DrakeMultibodyPlant = Union[
    MultibodyPlantFloat, MultibodyPlantAutoDiffXd, MultibodyPlantExpression
]

BodyFloat: TypeAlias = cast(Type, Body_[float])
BodyAutoDiffXd: TypeAlias = cast(Type, Body_[AutoDiffXd])
BodyExpression: TypeAlias = cast(Type, Body_[Expression])
DrakeBody = Union[BodyFloat, BodyAutoDiffXd, BodyExpression]

SpatialInertiaFloat: TypeAlias = cast(Type, SpatialInertia_[float])
SpatialInertiaAutoDiffXd: TypeAlias = cast(Type, SpatialInertia_[AutoDiffXd])
SpatialInertiaExpression: TypeAlias = cast(Type, SpatialInertia_[Expression])
DrakeSpatialInertia = Union[
    SpatialInertiaFloat, SpatialInertiaAutoDiffXd, SpatialInertiaExpression
]
#:
SceneGraphInspectorFloat: TypeAlias = cast(Type, SceneGraphInspector_[float])
SceneGraphInspectorAutoDiffXd: TypeAlias = cast(Type, SceneGraphInspector_[AutoDiffXd])
DrakeSceneGraphInspector = Union[
    SceneGraphInspectorFloat, SceneGraphInspectorAutoDiffXd
]
#:
DiagramBuilderFloat: TypeAlias = cast(Type, DiagramBuilder_[float])
DiagramBuilderAutoDiffXd: TypeAlias = cast(Type, DiagramBuilder_[AutoDiffXd])
DiagramBuilderExpression: TypeAlias = cast(Type, DiagramBuilder_[Expression])
DrakeDiagramBuilder = Union[
    DiagramBuilderFloat, DiagramBuilderAutoDiffXd, DiagramBuilderExpression
]
#:
UniqueBodyIdentifier = str


def get_bodies_in_model_instance(
    plant: DrakeMultibodyPlant, model_instance_index: ModelInstanceIndex
) -> List[DrakeBody]:
    """Get list of body names associated with model instance.

    Args:
        plant:
        model_instance_index:
    """
    body_indices = plant.GetBodyIndices(model_instance_index)
    return [plant.get_body(body_index) for body_index in body_indices]


def get_body_names_in_model_instance(
    plant: DrakeMultibodyPlant, model_instance_index: ModelInstanceIndex
) -> List[str]:
    """Get list of body names associated with model instance."""
    bodies = get_bodies_in_model_instance(plant, model_instance_index)
    return [body.name() for body in bodies]


def unique_body_identifier(
    plant: DrakeMultibodyPlant, body: DrakeBody
) -> UniqueBodyIdentifier:
    """Unique string identifier for given ``Body_``."""
    return f"{plant.GetModelInstanceName(body.model_instance())}_{body.name()}"


def get_body_from_geometry_id(
    plant: DrakeMultibodyPlant,
    inspector: DrakeSceneGraphInspector,
    geometry_id: GeometryId,
) -> DrakeBody:
    """Return the Drake Body from the given GeometryId"""
    frame_id = inspector.GetFrameId(geometry_id)
    return plant.GetBodyFromFrameId(frame_id)


def get_all_bodies(
    plant: DrakeMultibodyPlant, model_instance_indices: List[ModelInstanceIndex]
) -> Tuple[List[DrakeBody], List[UniqueBodyIdentifier]]:
    """Get all bodies in plant's models."""
    bodies = []
    for model_instance_index in model_instance_indices:
        bodies.extend(get_bodies_in_model_instance(plant, model_instance_index))
    return bodies, [unique_body_identifier(plant, body) for body in bodies]


def get_all_inertial_bodies(
    plant: DrakeMultibodyPlant, model_instance_indices: List[ModelInstanceIndex]
) -> Tuple[List[DrakeBody], List[UniqueBodyIdentifier]]:
    """Get all bodies that should have inertial parameters in plant."""
    bodies = []
    for model_instance_index in model_instance_indices:
        for body in get_bodies_in_model_instance(plant, model_instance_index):
            if not math.isnan(body.default_mass()) and body.default_mass() > 0.0:
                bodies.append(body)
    return bodies, [unique_body_identifier(plant, body) for body in bodies]


@dataclass
class CollisionGeometrySet:
    r""":py:func:`dataclasses.dataclass` for tracking object collisions."""

    ids: List[GeometryId] = field(default_factory=list)
    r"""List of geometries that may collide."""
    frictions: List[CoulombFriction] = field(default_factory=dict)  # type: ignore
    r"""List of coulomb friction coefficients for the geometries."""
    collision_candidates: List[Tuple[int, int]] = field(
        default_factory=dict
    )  # type: ignore
    r"""Pairs of geometries that may collide."""


def get_collision_geometry_set(
    inspector: DrakeSceneGraphInspector,
) -> CollisionGeometrySet:
    """Get colliding geometries, frictional properties, and corresponding
    collision pairs in a scene.

    Args:
        inspector: Inspector of scene graph.

    Returns:
        List of geometries that are candidates for at least one collision.
        Pairs of indices in geometry list that potentially collide.
    """
    geometry_ids: List[GeometryId] = []
    geometry_pairs: List[Tuple[int, int]] = []
    coulomb_frictions: List[CoulombFriction] = []

    for geometry_id_a, geometry_id_b in inspector.GetCollisionCandidates():
        for geometry_id in [geometry_id_a, geometry_id_b]:
            if geometry_id not in geometry_ids:
                geometry_ids.append(geometry_id)
        geometry_index_a = geometry_ids.index(geometry_id_a)
        geometry_index_b = geometry_ids.index(geometry_id_b)
        geometry_pairs.append((geometry_index_a, geometry_index_b))

    for geometry_id in geometry_ids:
        proximity_properties = inspector.GetProximityProperties(geometry_id)
        coulomb_frictions.append(
            proximity_properties.GetProperty(
                DRAKE_MATERIAL_GROUP, DRAKE_FRICTION_PROPERTY
            )
        )

    return CollisionGeometrySet(
        ids=geometry_ids,
        frictions=coulomb_frictions,
        collision_candidates=geometry_pairs,
    )


def add_plant_from_urdfs(
    builder: DrakeDiagramBuilder, urdfs: Dict[str, str], delta_t: float
) -> Tuple[List[ModelInstanceIndex], MultibodyPlant, SceneGraph]:
    """Add plant to builder with prescribed URDF models.

    Generates a world containing each given URDF as a model instance.

    Args:
        builder: Diagram builder to add plant to
        urdfs: Names and corresponding URDFs to add as models to plant.
        delta_t: Time step of plant in seconds.

    Returns:
        Named dictionary of model instances returned by
        ``AddModelsFromString``.
        New plant, which has been added to builder.
        Scene graph associated with new plant.
    """
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, delta_t)
    parser = Parser(plant)
    parser.SetAutoRenaming(True)

    # Build [model instance index] list, starting with world model, which is
    # always added by default.
    model_ids = [world_model_instance()]
    print(urdfs, "\n")
    for name, urdf in urdfs.items():
        print(urdf)
        new_ids = parser.AddModelsFromString(urdf, "urdf")
        if len(new_ids) < 1:
            continue
        assert len(new_ids) == 1, "Only one robot supported per URDF"
        model_ids.extend(new_ids)
        plant.RenameModelInstance(new_ids[0], name)

    return model_ids, plant, scene_graph


class ContactForceAveragerLeafSystem(LeafSystem):
    """Create a Drake ``LeafSystem`` which can average all point-point
    contact in the contact results between resets.
    """

    _average_index: int  # Abstract State Index
    _n_index: int  # Discrete State Index

    def __init__(self, contact_body_indices: List[BodyIndex], reset_on_output: bool = True):
        super().__init__()
        self._reset_on_output = reset_on_output

        # Create an input port for the current state of the system.
        self._contact_results_input_port = self.DeclareAbstractInputPort(
            "contact_results", Value(ContactResults())
        )

        self.DeclarePerStepUnrestrictedUpdateEvent(self.integrate_contact_results)
        forces_dict = {}
        point_normal_pair_dict = {}
        for body_idx in contact_body_indices:
            forces_dict[int(body_idx)] = np.zeros(3)
            point_normal_pair_dict[int(body_idx)] = [np.zeros(3), np.zeros(3)]
        self._average_index = self.DeclareAbstractState(
            Value(forces_dict)
        )  # Averaged Forces
        self._point_normal_index = self.DeclareAbstractState(
            Value(point_normal_pair_dict)
        )  # Averaged Forces
        self._n_index = self.DeclareDiscreteState(1)  # Number of samples in average

        self.DeclareAbstractOutputPort("averaged_contact_data",
                                       lambda: Value({"force": dict(),"point": dict(),"normal": dict()}),
                                       self.calc_average_output)

    def calc_average_output(self, context, forces_data):
        averages = self.get_value(context)
        point_normals = self.get_point_normals(context)
        for key, val in averages.items():
            forces_data.get_mutable_value()["force"][key] = val
            forces_data.get_mutable_value()["point"][key] = point_normals[key][0]
            forces_data.get_mutable_value()["normal"][key] = point_normals[key][1]

        if self._reset_on_output:
            self.reset(context)

    def get_point_normals(self, context):
        self.ValidateContext(context)
        return context.get_abstract_state(self._point_normal_index).get_value()

    def get_value(self, context):
        self.ValidateContext(context)
        return context.get_abstract_state(self._average_index).get_value()

    def reset(self, context):
        self.ValidateContext(context)

        old_n = context.get_mutable_discrete_state(self._n_index)
        old_n[0] = 0

        old_averages = context.get_mutable_abstract_state(
            self._average_index
        ).get_mutable_value()

        old_point_normals = context.get_mutable_abstract_state(
            self._point_normal_index
        ).get_mutable_value()

        for key in old_averages.keys():
            old_averages[key] = np.zeros(3)
            old_point_normals[key] = [np.zeros(3), np.zeros(3)]

    def integrate_contact_results(self, context, state_next):
        self.ValidateContext(context)
        # Evaluate the input ports to obtain the current multibody plant state.
        contact_results = self._contact_results_input_port.Eval(context)
        old_n = context.get_mutable_discrete_state(self._n_index)
        old_averages = context.get_mutable_abstract_state(
            self._average_index
        ).get_mutable_value()
        old_point_normals = context.get_mutable_abstract_state(
            self._point_normal_index
        ).get_mutable_value()
        new_val = {}
        for key in old_averages.keys():
            new_val[key] = np.zeros(3)

        for idx in range(contact_results.num_point_pair_contacts()):
            contact = contact_results.point_pair_contact_info(idx)
            a_idx = int(contact.bodyA_index())
            b_idx = int(contact.bodyB_index())
            new_val[a_idx] = new_val[a_idx] - contact.contact_force()
            new_val[b_idx] = new_val[b_idx] + contact.contact_force()

            old_point_normals[a_idx] = [contact.contact_point(), contact.point_pair().nhat_BA_W]
            old_point_normals[b_idx] = [contact.contact_point(), -contact.point_pair().nhat_BA_W]

        for key in old_averages.keys():
            old_averages[key] = (
                old_averages[key] * float(old_n[0]) + new_val[key]
            ) / float(old_n[0] + 1)

        # State updates
        old_n[0] = old_n[0] + 1
        state_next.get_mutable_discrete_state().set_value(np.array([old_n[0]]))
        state_next.get_mutable_abstract_state().get_mutable_value(self._average_index).SetFrom(Value(old_averages))
        state_next.get_mutable_abstract_state().get_mutable_value(self._point_normal_index).SetFrom(Value(old_point_normals))
        


@gin.configurable("PlantDiagram")
class MultibodyPlantDiagram:
    """Constructs and manages a diagram, simulator, and optionally a visualizer
    for a multibody system described in a list of URDF's.

    This minimal diagram consists of a ``MultibodyPlant``, ``SceneGraph``, and
    optionally a ``VideoWriter`` hooked up in the typical fashion.

    From the ``MultibodyPlant``, ``MultibodyPlantDiagram`` can infer the
    corresponding ``StateSpace`` from the dimension of the associated
    velocity vectors in the plant's context, via the one-chain-per-file
    assumption.
    """

    # pylint: disable=too-few-public-methods
    sim: Simulator
    plant: MultibodyPlant
    averager: ContactForceAveragerLeafSystem
    scene_graph: SceneGraph
    visualizer: Optional[Union[VideoWriter, MeshcatVisualizer]]
    model_ids: List[ModelInstanceIndex]
    collision_geometry_set: CollisionGeometrySet
    space: state_space.ProductSpace

    gin.constants_from_enum(cast(ContactModel, enum.Enum))
    gin.constants_from_enum(cast(DiscreteContactApproximation, enum.Enum))

    def __init__(
        self,
        urdfs: Dict[str, str],
        visualization_file: Optional[str] = None,
        additional_system_builders: Optional[
            List[Callable[[DrakeDiagramBuilder, MultibodyPlant], None]]
        ] = None,
        delta_t: float = SIM_DT,
        g_frac: Optional[float] = 1.0,
        ground_mu: Optional[float] = 1.0,
        contact_model: ContactModel = ContactModel.kPoint,
        contact_approx: DiscreteContactApproximation = DiscreteContactApproximation.kSap,
    ) -> None:
        r"""Initialization generates a world containing each given URDF as a
        model instance, and a corresponding Drake ``Simulator`` set up to
        trigger a state update every ``delta_t``.

        By default, a ground plane is added at world height ``z = 0``.

        Args:
            urdfs: Names and corresponding URDFs to add as models to plant.
            delta_t: Time step of plant in seconds.
            visualization_file: Optional output GIF filename for trajectory
              visualization.
            additional_system_builders: Optional functions that add additional Drake
              Systems to the plant diagram.
        """
        if additional_system_builders is None:
            additional_system_builders = []
        builder = DiagramBuilder()
        model_ids, plant, scene_graph = add_plant_from_urdfs(builder, urdfs, delta_t)

        # Add visualizer to diagram if enabled. Sets ``delete_prefix_on_load``
        # to False, in the hopes of saving computation time; may cause
        # re-initialization to produce erroneous visualizations.
        visualizer = None
        if visualization_file == "meshcat":
            self.meshcat = StartMeshcat()

            """ Useful for planar_xz
            ortho_camera = Meshcat.OrthographicCamera()
            ortho_camera.top = 1
            ortho_camera.bottom = -0.1
            ortho_camera.left = -1
            ortho_camera.right = 2
            ortho_camera.near = -10
            ortho_camera.far = 500
            ortho_camera.zoom = 1
            self.meshcat.SetCamera(ortho_camera)
            """
            visualizer = MeshcatVisualizer.AddToBuilder(
                builder, scene_graph, self.meshcat, params=MeshcatVisualizerParams(role=Role.kPerception)
            )
        elif visualization_file:
            visualizer = VideoWriter.AddToBuilder(
                filename=visualization_file,
                builder=builder,
                sensor_pose=SENSOR_POSE,
                fps=FPS,
                width=VIDEO_PIXELS[1],
                height=VIDEO_PIXELS[0],
                fov_y=CAM_FOV,
            )

        # Set contact model
        plant.set_contact_model(contact_model)
        plant.set_discrete_contact_approximation(contact_approx)

        # Adds ground plane at ``z = 0``

        halfspace_transform = RigidTransform_[float]()
        friction = CoulombFriction_[float](ground_mu, ground_mu)
        plant.RegisterCollisionGeometry(
            plant.world_body(),
            halfspace_transform,
            HalfSpace(),
            WORLD_GROUND_PLANE_NAME,
            friction,
        )
        plant.RegisterVisualGeometry(
            plant.world_body(),
            halfspace_transform,
            HalfSpace(),
            WORLD_GROUND_PLANE_NAME,
            GROUND_COLOR,
        )

        # get collision candidates before default context filters for proximity.
        inspector = scene_graph.model_inspector()
        self.collision_geometry_set = get_collision_geometry_set(inspector)

        # Edit the gravitational constant.
        new_gravity_vector = np.array([0.0, 0.0, -9.81 * g_frac])
        plant.mutable_gravity_field().set_gravity_vector(new_gravity_vector)

        # TODO: Document Weld World Frame
        for name in urdfs.keys():
            model = plant.GetModelInstanceByName(name)
            if plant.HasFrameNamed("worldfixed", model):
                plant.WeldFrames(
                    plant.world_frame(), plant.GetFrameByName("worldfixed", model)
                )

        # Gravcomp
        # TODO: this is a HACK, find principled way of doing gravcomp
        try:
            plant.set_gravity_enabled(plant.GetModelInstanceByName("robot"), False)
        except:
            print("Warning: No robot in plant")

        # Finalize multibody plant.
        plant.Finalize()

        # Create Contact Forces Averager
        contact_body_indices = [
            get_body_from_geometry_id(plant, inspector, geom_id).index()
            for geom_id in self.collision_geometry_set.ids
        ]
        self.averager = builder.AddNamedSystem(
            "averager",
            ContactForceAveragerLeafSystem(contact_body_indices)
        )
        builder.Connect(
            plant.get_contact_results_output_port(), self.averager.get_input_port()
        )

        # Call Additional System Builders
        for additional_system in additional_system_builders:
            additional_system(builder, plant)

        # Build diagram.
        diagram = builder.Build()
        diagram.CreateDefaultContext()

        # Uncomment the below lines to generate diagram graph.
        # diagram.set_name("graphviz example")
        # plt.figure(figsize=(11,8.5), dpi=300)
        # plot_system_graphviz(diagram)
        # from pathlib import Path
        # plt.savefig(str(Path.home() / "Desktop" / "plant_diagram.png"))
        # plt.close()

        # Initialize simulator from diagram.
        sim = Simulator(diagram)
        sim.set_publish_at_initialization(True)

        # Need constant visualization for Meshcat
        sim.set_publish_every_time_step(visualization_file == "meshcat")

        sim.Initialize()

        # Give use time to start Meshcat
        if visualization_file == "meshcat":
            input("Start Meshcat now!")

        # Ensure the model_ids order matches the state order.
        first_joint_indices = []
        for model_id in model_ids:
            if len(plant.GetJointIndices(model_id)) < 1:
                # Always put world model and weld joints first
                first_joint_indices.append(-1)
                continue
            first_joint_index = plant.GetJointIndices(model_id)
            int_index = int(str(first_joint_index).split("(")[1].split(")")[0])
            first_joint_indices.append(int_index)

        sorted_pairs = sorted(zip(first_joint_indices, model_ids))
        self.model_ids = [element for _, element in sorted_pairs]

        self.sim = sim
        self.diagram = diagram
        self.plant = plant
        self.scene_graph = scene_graph
        self.visualizer = visualizer
        self.space = self.generate_state_space()

    def vis_is_meshcat(self) -> bool:
        return isinstance(self.visualizer, MeshcatVisualizer)

    def generate_state_space(self) -> state_space.ProductSpace:
        """Generate ``StateSpace`` object for plant.

        Under the one-chain-per-model assumption, iteratively constructs a
        ``ProductSpace`` representation for the state of the ``MultibodyPlant``.

        Returns:
            State space of the diagram's underlying multibody system.
        """
        plant = self.plant

        spaces = []  # type: List[state_space.StateSpace]
        for model_id in self.model_ids:
            is_floating = False
            n_joints = 0
            for joint_idx in plant.GetJointIndices(model_id):
                joint = plant.get_joint(joint_idx)
                if joint.type_name() == "quaternion_floating":
                    is_floating = True
                elif joint.type_name() == "weld":
                    continue
                elif joint.type_name() in ["prismatic", "revolute"]:  # R^N
                    assert (
                        joint.num_positions() == joint.num_velocities()
                    ), f"Joint {joint.name()} not in R^N"
                    n_joints = n_joints + joint.num_positions()
                elif joint.type_name() in ["continuous", "planar"]:  # R^N x SO(2)
                    print(
                        f"WARNING: joint {joint.name()} of type {joint.type_name()} has an SO(2) DOF but will be modeled as R"
                    )
                    assert (
                        joint.num_positions() == joint.num_velocities()
                    ), f"Joint {joint.name()} not in R^N"
                    n_joints = n_joints + joint.num_positions()
                else:
                    raise ValueError(
                        f"Joint {joint.name()} of type {joint.type_name()} cannot be modeled in R^N or SE(3)"
                    )
            if is_floating:
                spaces.append(state_space.FloatingBaseSpace(n_joints))
            else:
                spaces.append(state_space.FixedBaseSpace(n_joints))

        return state_space.ProductSpace(spaces)
