"""Utility functions for generating URDF's for a given multibody system.

The ``URDFFindOrDefault`` class searches for elements in an urdf's xml tree,
and places a default in the event that the element does not exist. Many
string literals are instantiated here for convenience.

The ``UrdfGeometryRepresentationFactory`` generates URDF XML representations
of a ``CollisionGeometry``, and ``fill_link_with_parameterization`` dumps
these representations into a URDF "link" tag.
"""
import os.path
from typing import Dict, List, Optional, Tuple, cast
from xml.etree import ElementTree
from xml.etree.ElementTree import register_namespace

from torch import Tensor

from dair_pll import drake_utils
from dair_pll.deep_support_function import extract_obj
from dair_pll.geometry import CollisionGeometry, Box, Sphere, Polygon, \
    DeepSupportConvex
from dair_pll.inertia import InertialParameterConverter
from dair_pll.multibody_terms import MultibodyTerms

# tags
_ORIGIN = "origin"
_MASS = "mass"
_INERTIA = "inertia"
_INERTIAL = "inertial"
_VISUAL = "visual"
_COLLISION = "collision"
_GEOMETRY = "geometry"
_BOX = "box"
_SPHERE = "sphere"
_CYLINDER = "cylinder"
_MESH = "mesh"
_DRAKE = "{https://drake.mit.edu/}"
_PROXIMITY_PROPERTIES = "proximity_properties"
_DRAKE_PROXIMITY_PROPERTIES = _DRAKE + _PROXIMITY_PROPERTIES
_FRICTION = "mu_static"
_DRAKE_FRICTION = _DRAKE + _FRICTION

# attributes
_VALUE = "value"
_SIZE = "size"
_RADIUS = "radius"
_LENGTH = "length"
_FILENAME = "filename"
_XYZ = "xyz"
_RPY = "rpy"
_IXX = "ixx"
_IYY = "iyy"
_IZZ = "izz"
_IXY = "ixy"
_IXZ = "ixz"
_IYZ = "iyz"
_INERTIAL_ATTRIBUTES = [_IXX, _IYY, _IZZ, _IXY, _IXZ, _IYZ]

# values
_ZERO_FLOAT_3 = "0. 0. 0."
_ZERO_FLOAT = "0."

_POSE_ATTR = {_XYZ: _ZERO_FLOAT_3, _RPY: _ZERO_FLOAT_3}
_SCALAR_ATTR = {_VALUE: _ZERO_FLOAT}

_URDF_DEFAULT_TREE: Dict[str, List] = {
    _ORIGIN: [],
    _MASS: [],
    _INERTIA: [],
    _INERTIAL: [_ORIGIN, _MASS, _INERTIA],
    _GEOMETRY: [],
    _DRAKE_FRICTION: [],
    _DRAKE_PROXIMITY_PROPERTIES: [_DRAKE_FRICTION],
    _VISUAL: [_GEOMETRY, _ORIGIN],
    _COLLISION: [_GEOMETRY, _ORIGIN, _DRAKE_PROXIMITY_PROPERTIES],
    _BOX: [],
    _SPHERE: [],
    _CYLINDER: []
}  # pylint: disable=C0303
"""Default tree structure for URDF elements. 

Example:
    <inertial> elements contains <origin>, <mass>, and <inertia>, 
    sub-elements, thus::
        _URDF_DEFAULT_TREE[_INERTIAL] == [_ORIGIN, _MASS, _INERTIA]
"""

_URDF_DEFAULT_ATTRIBUTES: Dict[str, Dict] = {
    _ORIGIN: _POSE_ATTR,
    _MASS: _SCALAR_ATTR,
    _INERTIA: {i: _ZERO_FLOAT for i in _INERTIAL_ATTRIBUTES},
    _INERTIAL: {},
    _BOX: {
        _SIZE: _ZERO_FLOAT_3
    },
    _SPHERE: {
        _RADIUS: _ZERO_FLOAT
    },
    _CYLINDER: {
        _RADIUS: _ZERO_FLOAT,
        _LENGTH: _ZERO_FLOAT
    },
    _GEOMETRY: {},
    _VISUAL: {},
    _COLLISION: {},
    _DRAKE_PROXIMITY_PROPERTIES: {},
    _DRAKE_FRICTION: _SCALAR_ATTR
}
"""Default element attributes for URDFs.

Example:
    the <sphere> tag contains a "radius" parameter with float value, so::
        _URDF_DEFAULT_ATTRIBUTES[_SPHERE] == {_RADIUS: _ZERO_FLOAT}
"""


class UrdfFindOrDefault:
    """URDF XML tool to automatically fill in default element tree structures.

    URDF's often represent an identifiable unit (e.g. a body's spatial
    inertia) as a subtree of XML elements. ``URDFFindOrDefault`` implements a
    generalization of the ``xml.etree.ElementTree.find()`` method, which fills
    in a default subtree according to the tree structure given in
    ``_URDF_DEFAULT_TREE``, with each element given tags according to
    ``_URDF_DEFAULT_ATTRIBUTES``.

    Typical usage example::

        # element is an empty <inertial></inertial>
        # obtain default mass
        mass_element = URDFFindOrDefault.find(element, "mass")

        # element is now <inertial><mass value="0." /></inertial>
        # mass_element is now the child of element, <mass value="0." />

    """

    @staticmethod
    def find(element: ElementTree.Element,
             sub_element_type: str) -> ElementTree.Element:
        """Finds an XML sub-element of specified type, adding a default
        element of that type if necessary.

        Args:
            element: Element containing the sub-element.
            sub_element_type: Name of the sub-element type.

        Returns:
            An ``ElementTree.Element``, of type ``sub_element_type``, which is a
            child of the argument element that either (a) one which existed
            before the function call or (b) the root of a new, default subtree.

        Todo:
            * properly consider case where ``element`` already has multiple
        sub-elements of given type.
        """
        current_sub_element: Optional[ElementTree.Element] = element.find(
            sub_element_type)
        if current_sub_element is None:
            default_sub_element = \
                UrdfFindOrDefault.generate_default_element(sub_element_type)
            element.append(default_sub_element)
            return default_sub_element
        return current_sub_element

    @staticmethod
    def generate_default_element(element_type: str) -> ElementTree.Element:
        """Generates a default ``ElementTree.Element`` subtree of given type.

        Args:
            element_type: Name of the new default element type.

        Returns:
            A default ``ElementTree.Element`` of type ``element_type``.
        """
        default_element = ElementTree.Element(element_type)
        default_element.attrib = _URDF_DEFAULT_ATTRIBUTES[element_type]
        for child_element_type in _URDF_DEFAULT_TREE[element_type]:
            default_element.append(
                UrdfFindOrDefault.generate_default_element(child_element_type))
        return default_element


class UrdfGeometryRepresentationFactory:
    """Utility class for generating URDF representations of
    ``CollisionGeometry`` instances."""

    @staticmethod
    def representation(geometry: CollisionGeometry,
                       output_dir: str) -> Tuple[str, Dict[str, str]]:
        """Representation of an associated URDF tag that describes the
        properties of this geometry.

        Tags are expected to be put inside a ``<collision>`` tag in the URDF
        file.

        Example:
            To output ``<sphere radius="5.1">`` for a ``Sphere``, return the
            following::

                ('sphere', {'radius': '5.1'})
        Args:
            geometry: collision geometry to be represented
            output_dir: File directory to store helper files (e.g., meshes).

        Returns:
            URDF tag and attributes.
        """
        if isinstance(geometry, Polygon):
            return UrdfGeometryRepresentationFactory.polygon_representation()
        if isinstance(geometry, Box):
            return UrdfGeometryRepresentationFactory.box_representation(
                geometry)
        if isinstance(geometry, Sphere):
            return UrdfGeometryRepresentationFactory.sphere_representation(
                geometry)
        if isinstance(geometry, DeepSupportConvex):
            return UrdfGeometryRepresentationFactory.mesh_representation(
                geometry, output_dir)
        raise TypeError(
            "Unsupported type for CollisionGeometry() to"
            "URDF representation conversion:", type(geometry))

    @staticmethod
    def polygon_representation() -> Tuple[str, Dict[str, str]]:
        """Todo: implement representation for ``Polygon``"""
        raise NotImplementedError("Polygon URDF representation not yet "
                                  "implemented.")

    @staticmethod
    def box_representation(box: Box) -> Tuple[str, Dict[str, str]]:
        """Returns URDF representation as ``box`` tag with full-length sizes."""
        size = ' '.join([str(2 * i.item()) for i in box.half_lengths.view(-1)])
        return _BOX, {_SIZE: size}

    @staticmethod
    def sphere_representation(sphere: Sphere) -> Tuple[str, Dict[str, str]]:
        """Returns URDF representation as ``sphere`` tag with radius
        attribute."""
        return _SPHERE, {_RADIUS: str(sphere.radius.item())}

    @staticmethod
    def mesh_representation(convex: DeepSupportConvex, output_dir: str) -> \
            Tuple[str, Dict[str, str]]:
        """Returns URDF representation as ``mesh`` tag with name of saved
        mesh file."""
        #pdb.set_trace()
        mesh_name = "test.obj"
        mesh_path = os.path.join(output_dir, mesh_name)
        #print(mesh_path)
        with open(mesh_path, 'w', encoding="utf8") as new_obj_file:
            new_obj_file.write(extract_obj(convex.network))

        return _MESH, {_FILENAME: mesh_name}


def fill_link_with_parameterization(element: ElementTree.Element, pi_cm: Tensor,
                                    geometries: List[CollisionGeometry],
                                    frictions: Tensor,
                                    output_dir: str) -> None:
    """Convert pytorch inertial and geometric representations to URDF elements.

    Args:
        element: XML "link" tag in which representation is stored.
        pi_cm: (10,) inertial representation of link in ``pi_cm``
                parameterization.
        geometries: All geometries attached to body.
        frictions: All friction coefficients associated with each geometry.  The
          ``Tensor`` will be of shape ``(len(geometries),)``.
        output_dir: File directory to store helper files (e.g., meshes).

    Warning:
        Does not handle multiple geometries.
    Todo:
        Handle multiple geometries for body.
    Raises:
        NotImplementedError: when multiple geometries are provided.
    """
    if len(geometries) > 1:
        raise NotImplementedError("generating a URDF with multiple geometries"
                                  "per body not implemented yet.")
    mass, p_BoBcm_B, I_BBcm_B = \
        InertialParameterConverter.pi_cm_to_urdf(pi_cm)

    # This will have to change when function can handle more than one geometry.
    mu = str(frictions.item())

    body_inertial_element = UrdfFindOrDefault.find(element, _INERTIAL)

    UrdfFindOrDefault.find(body_inertial_element, _MASS).set(_VALUE, mass)
    UrdfFindOrDefault.find(body_inertial_element, _ORIGIN).set(_XYZ, p_BoBcm_B)

    body_inertia_element = UrdfFindOrDefault.find(body_inertial_element,
                                                  _INERTIA)
    body_inertia_element.attrib = dict(zip(_INERTIAL_ATTRIBUTES, I_BBcm_B))

    for geometry in geometries:
        collision_element = UrdfFindOrDefault.find(element, _COLLISION)
        visual_element = UrdfFindOrDefault.find(element, _VISUAL)
        geometry_elements = [
            UrdfFindOrDefault.find(collision_element, _GEOMETRY),
            UrdfFindOrDefault.find(visual_element, _GEOMETRY)
        ]
        
        (shape_tag, shape_attributes) = \
            UrdfGeometryRepresentationFactory.representation(geometry,
                                                             output_dir)
        for geometry_element in geometry_elements:
            shape_element = UrdfFindOrDefault.find(geometry_element, shape_tag)
            shape_element.attrib = shape_attributes

        prox_props_element = UrdfFindOrDefault.find(collision_element,
            _DRAKE_PROXIMITY_PROPERTIES)
        UrdfFindOrDefault.find(prox_props_element, _DRAKE_FRICTION).set(
            _VALUE, mu)


def represent_multibody_terms_as_urdfs(multibody_terms: MultibodyTerms,
                                       output_dir: str) -> Dict[str, str]:
    """Renders the current parameterization of multibody terms as a
    set of urdfs.

    Args:
        multibody_terms: Multibody dynamics representation to convert.
        output_dir: File directory to store helper files (e.g., meshes).
    Returns:
        Dictionary of (urdf name, urdf XML string) pairs.
    Warning:
        For now, assumes that each URDF link element ``e`` gets modeled as a
        corresponding body ``b`` with ``b.name() == e.get("name")``.
        Drake however does not guarantee this relationship. A more stable
        implementation would be to directly edit the MultibodyPlant, but this
        would make the representation less portable.
    """
    # pylint: disable=too-many-locals
    urdf_xml = {}
    _, all_body_ids = \
        drake_utils.get_all_inertial_bodies(
            multibody_terms.plant_diagram.plant,
            multibody_terms.plant_diagram.model_ids)
    pi_cm = multibody_terms.lagrangian_terms.pi_cm()

    for urdf_name, urdf in multibody_terms.urdfs.items():

        # assumes urdf name mirrors model name
        model_instance_index = \
            multibody_terms.plant_diagram.plant.GetModelInstanceByName(
                urdf_name)

        urdf_tree = ElementTree.parse(urdf)

        for element in urdf_tree.iter():
            if element.tag == "link":
                body_id = drake_utils.unique_body_identifier(
                    multibody_terms.plant_diagram.plant,
                    multibody_terms.plant_diagram.plant.GetBodyByName(
                        element.get("name"), model_instance_index))
                if body_id not in all_body_ids:
                    # body does not have inertial attributes,
                    # for instance, the world body.
                    continue
                body_index = all_body_ids.index(body_id)
                body_geometry_indices = \
                    multibody_terms.geometry_body_assignment[body_id]
                body_geometries = [
                    cast(CollisionGeometry,
                         multibody_terms.contact_terms.geometries[index])
                    for index in body_geometry_indices
                ]
                body_frictions = \
                    multibody_terms.contact_terms.friction_coefficients[
                        body_geometry_indices
                    ]
                fill_link_with_parameterization(element, pi_cm[body_index, :],
                                                body_geometries, body_frictions,
                                                output_dir)

        register_namespace('drake', 'https://drake.mit.edu/')
        system_urdf_representation = ElementTree.tostring(
            urdf_tree.getroot(), encoding="utf-8").decode("utf-8")
        urdf_xml[
            urdf_name] = f'<?xml version="1.0"?>\n{system_urdf_representation}'
    return urdf_xml
