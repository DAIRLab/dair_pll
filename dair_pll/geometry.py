"""Collision geometry representation for multibody systems.

Each type of collision geometry is modeled as a class inheriting from the
``CollisionGeometry`` abstract type. Different types of inheriting geometries
will need to resolve collisions in unique ways, but one interface is always
expected: a list of scalars to track during training.

Many collision geometries, such as boxes and cylinders, fall into the class
of bounded (compact) convex shapes. A general interface is defined in the
abstract ``BoundedConvexCollisionGeometry`` type, which returns a set of
witness points given support hyperplane directions. One implementation is
the ``SparseVertexConvexCollisionGeometry`` type, which finds these points
via brute force optimization over a short list of vertices.

All collision geometries implemented here mirror a Drake ``Shape`` object. A
general purpose converter is implemented in
``PydrakeToCollisionGeometryFactory``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Dict, cast, Union

import fcl  # type: ignore
import numpy as np
import pywavefront  # type: ignore
import torch
from pydrake.geometry import Box as DrakeBox  # type: ignore
from pydrake.geometry import Sphere as DrakeSphere # type: ignore
from pydrake.geometry import HalfSpace as DrakeHalfSpace  # type: ignore
from pydrake.geometry import Mesh as DrakeMesh  # type: ignore
from pydrake.geometry import Shape  # type: ignore
from torch import Tensor
from torch.nn import Module, Parameter

from dair_pll.deep_support_function import HomogeneousICNN, \
    extract_mesh_from_support_function
from dair_pll.tensor_utils import pbmm, tile_dim, \
    rotation_matrix_from_one_vector

_UNIT_BOX_VERTICES = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1.], [
    0, 0, 1, 1, 0, 0, 1, 1.
], [0, 1, 0, 1, 0, 1, 0, 1.]]).t() * 2. - 1.

_ROT_Z_45 = torch.tensor([[2**(-0.5), -(2**(-0.5)), 0.], [2**(-0.5), 2**(-0.5), 0.],
                    [0., 0., 1.]])

_NOMINAL_HALF_LENGTH = 1.0   # Note: matches Box/Polygon space to trajectory space (m)

_total_ordering = ['Plane', 'Polygon', 'Box', 'Sphere', 'DeepSupportConvex']

_POLYGON_DEFAULT_N_QUERY = 4
_DEEP_SUPPORT_DEFAULT_N_QUERY = 4
_DEEP_SUPPORT_DEFAULT_DEPTH = 2
_DEEP_SUPPORT_DEFAULT_WIDTH = 256



class CollisionGeometry(ABC, Module):
    """Abstract interface for collision geometries.

    Collision geometries have heterogeneous implementation depending on the
    type of shape. This class mainly enforces the implementation of
    bookkeeping interfaces.

    When two collision geometries are evaluated for collision with
    ``GeometryCollider``, their ordering is constrained via a total order on
    collision geometry types, enforced with an overload of the ``>`` operator.
    """

    name: str = ""
    learnable: bool = False

    def __ge__(self, other) -> bool:
        """Evaluate total ordering of two geometries based on their types."""
        return _total_ordering.index(
            type(self).__name__) > _total_ordering.index(type(other).__name__)

    def __lt__(self, other) -> bool:
        """Evaluate total ordering of two geometries via passthrough to
        ``CollisionGeometry.__ge__()``."""
        return other.__ge__(self)

    @abstractmethod
    def scalars(self) -> Dict[str, float]:
        """Describes object via Tensorboard scalars.

        Any namespace for the object (e.g. "object_5") is assumed to be added by
        external code before adding to Tensorboard, so the names of the
        scalars can be natural descriptions of the geometry's parameters.

        Examples:
            A cylinder might be represented with the following output::

                {'radius': 5.2, 'height': 4.1}

        Returns:
            A dictionary of named parameters describing the geometry.
        """


class Plane(CollisionGeometry):
    """Half-space/plane collision geometry.

    ``Plane`` geometries are assumed to be the plane z=0 in local (i.e.
    "body-axes" coordinates). Any tilted/raised/lowered half spaces are expected
    to be derived by placing the ``z=0`` plane in a rigidly-transformed
    frame."""

    def scalars(self) -> Dict[str, float]:
        """As the plane is fixed to be z=0, there are no parameters."""
        return {}


class BoundedConvexCollisionGeometry(CollisionGeometry):
    """Interface for compact-convex collision geometries.

    Such shapes have a representation via a "support function" h(d),
    which takes in a hyperplane direction and returns how far the shape S
    extends in that direction, i.e.::

        h(d) = max_{s \\in S} s \\cdot d.

    This representation can be leveraged to derive "witness points" -- i.e.
    the closest point(s) between the ``BoundedConvexCollisionGeometry`` and
    another convex shape, such as another ``BoundedConvexCollisionGeometry``
    or a ``Plane``.
    """

    @abstractmethod
    def support_points(self, directions: Tensor, hint: Optional[Tensor] = None) -> Tensor:
        """Returns a set of witness points representing contact with another
        shape off in the direction(s) ``directions``.

        This method will return a set of points ``S' \\subset S`` such that::

            argmax_{s \\in S} s \\cdot directions \\subset convexHull(S').


        In theory, returning exactly the argmax set would be sufficient.
        However,

        Args:
            directions: (\*, 3) batch of unit-length directions.
            hint: (\*, 3) batch of expected contact point

        Returns:
            (\*, N, 3) sets of corresponding witness points of cardinality N.
            If hint is defined, then N == 1.
        """

    @abstractmethod
    def get_fcl_geometry(self) -> fcl.CollisionGeometry:
        """Retrieves :py:mod:`fcl` collision geometry representation.

        Returns:
            :py:mod:`fcl` bounding volume
        """


class SparseVertexConvexCollisionGeometry(BoundedConvexCollisionGeometry):
    """Partial implementation of ``BoundedConvexCollisionGeometry`` when
    witness points are guaranteed to be contained in a small set of vertices.

    An obvious subtype is any sort of polytope, such as a ``Box``. A less
    obvious subtype are shapes in which a direction-dependent set of vertices
    can be easily calculated. See ``Cylinder``, for instance.
    """

    def __init__(self, n_query: int) -> None:
        """Inits ``SparseVertexConvexCollisionGeometry`` with prescribed
        query interface.

        Args:
            n_query: number of vertices to return in witness point set.
        """
        super().__init__()
        self.n_query = n_query

    def support_points(self, directions: Tensor, hint: Optional[Tensor] = None) -> Tensor:
        """Implements ``BoundedConvexCollisionGeometry.support_points()`` via
        brute force optimization over the witness vertex set.

        Specifically, if S_v is the vertex set, this function returns
        ``n_query`` elements s of S_v for which ``s \\cdot directions`` is
        highest. This set is not guaranteed to be sorted.

        Given the prescribed behavior of
        ``BoundedConvexCollisionGeometry.support_points()``, an implicit
        assumption of this implementation is that the convex hull of the top
        ``n_query`` points in S_v contains ``argmax_{s \\in S} s \\cdot
        directions``.

        Args:
            directions: (\*, 3) batch of directions
            hint: (\*, 3) expected contact point, should be on convex set
            of the witness points. Used if n_query > 1

        Returns:
            (\*, n_query, 3) sets of corresponding witness points.
        """
        assert directions.shape[-1] == 3
        original_shape = directions.shape

        # reshape to (product(*),3)
        directions = directions.view(-1, 3)

        # pylint: disable=E1103
        batch_range = torch.arange(directions.shape[0])
        vertices = self.get_vertices(directions)
        dots = pbmm(directions.unsqueeze(-2),
                    vertices.transpose(-1, -2)).squeeze(-2)

        # top dot product indices in shape (product(*), n_query)
        # pylint: disable=E1103
        selections = torch.topk(dots, self.n_query, dim=-1,
                                sorted=True).indices.t()

        top_vertices = torch.stack(
            [vertices[batch_range, selection] for selection in selections], -2)
        # reshape to (*, n_query, 3)
        queries = top_vertices.view(original_shape[:-1] + (self.n_query, 3))
        if self.n_query > 1 and (hint is not None) and hint.shape == directions.shape:
            # Find linear combination of queries 
            # Lst Sq: queries (*, 3, n_query) * ? (*, n_query, 1) == hint (*, 1, 3)
            # Note: solution needs to be detached from the gradient chain.
            sol = torch.linalg.lstsq(queries.detach().transpose(-1, -2), hint.unsqueeze(-1)).solution
            return pbmm(queries.transpose(-1, -2), sol).transpose(-1, -2)

        return queries

    @abstractmethod
    def get_vertices(self, directions: Tensor) -> Tensor:
        """Returns sparse witness point set as collection of vertices.

        Specifically, given search directions, returns a set of points
        ``S_v`` for which::

            argmax_{s \\in S} s \\cdot directions \\subset convexHull(S_v).

        Args:
            directions: (\*, 3) batch of unit-length directions.
        Returns:
            (\*, N, 3) witness point convex hull vertices.
        """


class Polygon(SparseVertexConvexCollisionGeometry):
    """Concrete implementation of a convex polytope.

    Implemented via ``SparseVertexConvexCollisionGeometry`` as a static set
    of vertices, where models the underlying shape as all convex combinations
    of the vertices.
    """
    vertices_parameter: Parameter

    def __init__(self,
                 vertices: Tensor,
                 n_query: int = _POLYGON_DEFAULT_N_QUERY,
                 learnable: bool = True) -> None:
        """Inits ``Polygon`` object with initial vertex set.

        Args:
            vertices: (N, 3) static vertex set.
            n_query: number of vertices to return in witness point set.
        """
        super().__init__(n_query)
        scaled_vertices = vertices.clone()/_NOMINAL_HALF_LENGTH
        self.vertices_parameter = Parameter(scaled_vertices, requires_grad=learnable)
        self.learnable = learnable

    def get_vertices(self, directions: Tensor) -> Tensor:
        """Return batched view of static vertex set"""
        scaled_vertices = _NOMINAL_HALF_LENGTH * self.vertices_parameter
        return scaled_vertices.expand(
            directions.shape[:-1] + scaled_vertices.shape)

    def scalars(self) -> Dict[str, float]:
        """Return one scalar for each vertex index."""
        scalars = {}
        axes = ['x', 'y', 'z']

        # Use arbitrary direction to query the Polygon's vertices (value does
        # not matter).
        arbitrary_direction = torch.ones((1,3))
        vertices = self.get_vertices(arbitrary_direction).squeeze(0)

        for axis, values in zip(axes, vertices.t()):
            for vertex_index, value in enumerate(values):
                scalars[f'v{vertex_index}_{axis}'] = value.item()
        return scalars


class DeepSupportConvex(SparseVertexConvexCollisionGeometry):
    r"""Deep support function convex shape.

    Any convex shape :math:`S` can be equivalently represented via its support
    function :math:`f(d)`, which returns the extent to which the object
    extends in the :math:`d` direction:

    .. math::

        f(d) = \max_{s \in S} s \cdot d.

    Given a direction, the set of points that form the :math:`\arg\max` in
    :math:`f(d)` is exactly the convex subgradient :math:`\partial_d f(d)`.

    Furthermore, for every convex shape, :math:`f(d)` is convex and
    positively homogeneous, and every convex and positively homogeneous
    :math:`f(d)` is the support function of some convex shape.

    This collision geometry type implements the support function directly as
    a convex and positively homogeneous neural network (
    :py:class:`~dair_pll.deep_support_function.HomogeneousICNN`\)."""
    network: HomogeneousICNN
    """Support function representation as a neural net."""
    perturbations: Tensor
    """Perturbed support directions, which aid mesh-plane contact stability."""
    fcl_geometry: fcl.BVHModel
    r""":py:mod:`fcl` mesh collision geometry representation."""

    def __init__(self,
                 vertices: Tensor,
                 n_query: int = _DEEP_SUPPORT_DEFAULT_N_QUERY,
                 depth: int = _DEEP_SUPPORT_DEFAULT_DEPTH,
                 width: int = _DEEP_SUPPORT_DEFAULT_WIDTH,
                 perturbation: float = 0.4,
                 learnable: bool = True) -> None:
        r"""Inits ``DeepSupportConvex`` object with initial vertex set.

        When calculating a sparse vertex set with :py:meth:`get_vertices`,
        supplements the support direction with nearby directions randomly.

        Args:
            vertices: ``(N, 3)`` initial vertex set.
            n_query: Number of vertices to return in witness point set.
            depth: Depth of support function network.
            width: Width of support function network.
            perturbation: support direction sampling parameter.
        """
        # pylint: disable=too-many-arguments,E1103
        super().__init__(n_query)
        length_scale = (vertices.max(dim=0).values -
                        vertices.min(dim=0).values).norm() / 2
        self.network = HomogeneousICNN(depth, width, scale=length_scale, learnable=learnable)
        self.perturbations = torch.cat((torch.zeros(
            (1, 3)), perturbation * (torch.rand((n_query - 1, 3)) - 0.5)))
        self.learnable = learnable

    def get_vertices(self, directions: Tensor) -> Tensor:
        """Return batched view of support points of interest.

        Given a direction :math:`d`, this function finds the support point of
        the object in that direction, calculated via envelope

        Args:
            directions: ``(*, 3)`` batch of support directions sample.

        Returns:
            ``(*, n_query, 3)`` sampled support points.
        """
        perturbed = directions.unsqueeze(-2)
        perturbed = tile_dim(perturbed, self.n_query, -2)
        perturbed += self.perturbations.expand(perturbed.shape)
        perturbed /= perturbed.norm(dim=-1, keepdim=True)
        return self.network(perturbed)

    def train(self, mode: bool = True) -> DeepSupportConvex:
        r"""Override training-mode setter from :py:mod:`torch`.

        Sets a static fcl mesh geometry for the entirety of evaluation time,
        as the underlying support function is not changing.

        Args:
            mode: ``True`` for training, ``False`` for evaluation.

        Returns:
            ``self``.
        """
        if not mode:
            self.fcl_geometry = self.get_fcl_geometry()
        return cast(DeepSupportConvex, super().train(mode))

    def get_fcl_geometry(self) -> fcl.CollisionGeometry:
        """Retrieves :py:mod:`fcl` mesh collision geometry representation.

        If evaluation mode is set, retrieves precalculated version.

        Returns:
            :py:mod:`fcl` bounding volume hierarchy for mesh.
        """
        if self.training:
            mesh = extract_mesh_from_support_function(self.network)
            vertices = mesh.vertices.numpy()
            faces = mesh.faces.numpy()
            self.fcl_geometry = fcl.BVHModel()
            self.fcl_geometry.beginModel(vertices.shape[0], faces.shape[0])
            self.fcl_geometry.addSubModel(vertices, faces)
            self.fcl_geometry.endModel()

        return self.fcl_geometry

    def scalars(self) -> Dict[str, float]:
        """no scalars!"""
        return {}


class Box(SparseVertexConvexCollisionGeometry):
    """Implementation of cuboid geometry as a sparse vertex convex hull.

    To prevent learning negative box lengths, the learned parameters are stored
    as :py:attr:`length_params`, and the box's half lengths can be computed
    as their absolute value.  The desired half lengths can be accessed via
    :py:meth:`get_half_lengths`.
    """
    length_params: Parameter
    unit_vertices: Tensor

    def __init__(self, half_lengths: Tensor, n_query: int, learnable: bool = True) -> None:
        """Inits ``Box`` object with initial size.

        Args:
            half_lengths: (3,) half-length dimensions of box on x, y,
              and z axes.
            n_query: number of vertices to return in witness point set.
        """
        super().__init__(n_query)

        assert half_lengths.numel() == 3

        scaled_half_lengths = half_lengths.clone()/_NOMINAL_HALF_LENGTH
        self.length_params = Parameter(scaled_half_lengths.view(1, -1),
                                       requires_grad=learnable)
        self.length_params.register_hook(lambda grad: print(f"Box Param Gradient: {grad}"))
        self.unit_vertices = _UNIT_BOX_VERTICES.clone().to(device=self.length_params.device)
        self.learnable = learnable

    def get_half_lengths(self) -> Tensor:
        """From the stored :py:attr:`length_params`, compute the half lengths of
        the box as its absolute value."""
        return torch.abs(self.length_params) * _NOMINAL_HALF_LENGTH

    def get_vertices(self, directions: Tensor) -> Tensor:
        """Returns view of cuboid's static vertex set."""
        return (self.unit_vertices *
                self.get_half_lengths()).expand(directions.shape[:-1] +
                                                self.unit_vertices.shape)

    def scalars(self) -> Dict[str, float]:
        """Returns each axis's full length as a scalar."""
        scalars = {
            f'len_{axis}': 2 * value.item()
            for axis, value in zip(['x', 'y', 'z'],
                self.get_half_lengths().view(-1))
        }
        return scalars

    def get_fcl_geometry(self) -> fcl.CollisionGeometry:
        """Retrieves :py:mod:`fcl` collision geometry representation.

        Returns:
            :py:mod:`fcl` bounding volume
        """
        scalars = self.scalars()

        return fcl.Box(scalars["len_x"], scalars["len_y"], scalars["len_z"])


class Sphere(BoundedConvexCollisionGeometry):
    """Implements sphere geometry via its support function.

    It is trivial to calculate the witness point for a sphere contact as
    simply the product of the sphere's radius and the support direction.

    To prevent learning a negative radius, the learned parameter is stored as
    :py:attr:`length_param`, and the sphere's radius can be computed as its
    absolute value.  The desired radius can be accessed via
    :py:meth:`get_radius`.
    """
    length_param: Parameter

    def __init__(self, radius: Tensor, learnable: bool = True) -> None:
        super().__init__()
        assert radius.numel() == 1

        self.length_param = Parameter(radius.clone().view(()),
                                      requires_grad=learnable)
        self.learnable = learnable

    def get_radius(self) -> Tensor:
        """From the stored :py:attr:`length_param`, compute the radius of the
        sphere as its absolute value."""
        return torch.abs(self.length_param)

    def support_points(self, directions: Tensor, _: Optional[Tensor] = None) -> Tensor:
        """Implements ``BoundedConvexCollisionGeometry.support_points()``
        via analytic expression::

            argmax_{s \\in S} s \\cdot directions = directions * radius.

        Args:
            directions: (\*, 3) batch of directions.

        Returns:
            (\*, 1, 3) corresponding witness point sets of cardinality 1.
        """
        return (directions.clone() * self.get_radius()).unsqueeze(-2)

    def scalars(self) -> Dict[str, float]:
        """Logs radius as a scalar."""
        return {'radius': self.get_radius().item()}

    def get_fcl_geometry(self) -> fcl.CollisionGeometry:
        """Retrieves :py:mod:`fcl` collision geometry representation.

        Returns:
            :py:mod:`fcl` bounding volume
        """
        scalars = self.scalars()

        return fcl.Sphere(scalars["radius"])


class PydrakeToCollisionGeometryFactory:
    """Utility class for converting Drake ``Shape`` instances to
    ``CollisionGeometry`` instances."""

    @staticmethod
    def convert(drake_shape: Shape, represent_geometry_as: str,
        learnable: bool = True, name: str = ""
        ) -> CollisionGeometry:
        """Converts abstract ``pydrake.geometry.shape`` to
        ``CollisionGeometry`` according to the desired ``represent_geometry_as``
        type.

        Notes:
            The desired ``represent_geometry_as`` type only will affect
            ``DrakeBox`` and ``DrakeMesh`` types, not ``DrakeHalfSpace`` types.

        Args:
            drake_shape: drake shape type to convert.

        Returns:
            Collision geometry representation of shape.

        Raises:
            TypeError: When provided object is not a supported Drake shape type.
        """
        if isinstance(drake_shape, DrakeBox):
            geometry = PydrakeToCollisionGeometryFactory.convert_box(
                drake_shape, represent_geometry_as, learnable)
        elif isinstance(drake_shape, DrakeHalfSpace):
            geometry = PydrakeToCollisionGeometryFactory.convert_plane()
        elif isinstance(drake_shape, DrakeMesh):
            geometry = PydrakeToCollisionGeometryFactory.convert_mesh(
                drake_shape, represent_geometry_as, learnable)
        elif isinstance(drake_shape, DrakeSphere):
            geometry = PydrakeToCollisionGeometryFactory.convert_sphere(
                drake_shape, represent_geometry_as, learnable)
        else:
            raise TypeError(
                "Unsupported type for drake Shape() to"
                "CollisionGeometry() conversion:", type(drake_shape))

        geometry.name = name
        return geometry

    @staticmethod
    def convert_box(drake_box: DrakeBox, represent_geometry_as: str, learnable: bool = True
        ) -> Union[Box, Polygon]:
        """Converts ``pydrake.geometry.Box`` to ``Box`` or ``Polygon``."""
        if represent_geometry_as == 'box':
            half_widths = 0.5 * torch.tensor(np.copy(drake_box.size()))
            return Box(half_widths, 4, learnable)

        if represent_geometry_as == 'polygon':
            pass # TODO

        raise NotImplementedError(f'Cannot presently represent a DrakeBox()' + \
            f'as {represent_geometry_as} type.')

    @staticmethod
    def convert_sphere(drake_sphere: DrakeSphere, represent_geometry_as: str, learnable: bool = True
        ) -> Union[Sphere, Polygon]:
        """Converts ``pydrake.geometry.Box`` to ``Box`` or ``Polygon``."""
        if represent_geometry_as == 'box':
            return Sphere(torch.tensor([drake_sphere.radius()]), learnable)

        if represent_geometry_as == 'polygon':
            pass # TODO

        raise NotImplementedError(f'Cannot presently represent a DrakeBox()' + \
            f'as {represent_geometry_as} type.')

    @staticmethod
    def convert_plane() -> Plane:
        """Converts ``pydrake.geometry.HalfSpace`` to ``Plane``."""
        return Plane()

    @staticmethod
    def convert_mesh(drake_mesh: DrakeMesh, represent_geometry_as: str, learnable: bool = True
        ) -> Union[DeepSupportConvex, Polygon]:
        """Converts ``pydrake.geometry.Mesh`` to ``Polygon`` or
        ``DeepSupportConvex``."""
        filename = drake_mesh.filename()
        mesh = pywavefront.Wavefront(filename)
        vertices = torch.tensor(mesh.vertices)

        if represent_geometry_as == 'mesh':
            return DeepSupportConvex(vertices, learnable=learnable)

        if represent_geometry_as == 'polygon':
            return Polygon(vertices, learnable)

        raise NotImplementedError(f'Cannot presently represent a ' + \
            f'DrakeMesh() as {represent_geometry_as} type.')


class GeometryCollider:
    """Utility class for colliding two ``CollisionGeometry`` instances."""

    @staticmethod
    def collide(geometry_a: CollisionGeometry, geometry_b: CollisionGeometry,
                R_AB: Tensor, p_AoBo_A: Tensor, estimated_normals_A: Optional[Tensor]) -> \
            Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Collides two collision geometries.

        Takes in the two geometries as well as a relative transform between
        them. This function is a thin shell for other static methods of
        ``GeometryCollider`` where the given geometries are guaranteed to
        have specific types.

        Args:
            geometry_a: first collision geometry
            geometry_b: second collision geometry, with type
              ordering ``not geometry_A > geometry_B``.
            R_AB: (\*,3,3) rotation between geometry frames
            p_AoBo_A: (\*, 3) offset of geometry frame origins
            estimated_normals_A: (\*, 3) estimate of contact normal from A

        Returns:
            (\*, N) batch of witness point pair distances
            (\*, N, 3, 3) contact frame C rotation in A, R_AC, where the z
            axis of C is contained in the normal cone of body A at contact
            point Ac and is parallel (or antiparallel) to AcBc.
            (\*, N, 3) witness points Ac on A, p_AoAc_A
            (\*, N, 3) witness points Bc on B, p_BoBc_B
        """
        assert not geometry_a > geometry_b

        # case 1: half-space to compact-convex collision
        # TODO: make function allow planes in general, not just in 1st slot
        if isinstance(geometry_a, Plane) and isinstance(
                geometry_b, BoundedConvexCollisionGeometry):
            return GeometryCollider.collide_plane_convex(
                geometry_b, R_AB, p_AoBo_A)
        if isinstance(geometry_b, Plane) and isinstance(
                geometry_a, BoundedConvexCollisionGeometry):
            return GeometryCollider.collide_plane_convex(
                geometry_a, R_AB.transpose(-1, -2), -pbmm(p_AoBo_A, R_AB))
        if isinstance(geometry_a, Box) and isinstance(
                geometry_b, Sphere):
            return GeometryCollider.collide_box_sphere(
                geometry_a, geometry_b, R_AB, p_AoBo_A, estimated_normals_A)
        if isinstance(geometry_a, Sphere) and isinstance(
                geometry_b, Box):
            return GeometryCollider.collide_box_sphere(
                geometry_b, geometry_a, R_AB.transpose(-1, -2), -pbmm(p_AoBo_A, R_AB), -pbmm(estimated_normals_A, R_AB))
        if isinstance(geometry_a, BoundedConvexCollisionGeometry) and isinstance(
                geometry_b, BoundedConvexCollisionGeometry):
            return GeometryCollider.collide_convex_convex(geometry_a, geometry_b,
                                                      R_AB, p_AoBo_A)
        raise TypeError(
            "No type-specific implementation for geometry "
            "pair of following types:",
            type(geometry_a).__name__,
            type(geometry_b).__name__)

    @staticmethod
    def collide_box_sphere(box_a: Box, sphere_b: sphere,
                             R_AB: Tensor, p_AoBo_A: Tensor, estimated_normals_A: Optional[Tensor]) -> \
            Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Implementation of ``GeometryCollider.collide()`` when
        ``geometry_a`` is a ``Box`` and ``geometry_b`` is a
        ``Sphere``.

        box_a: Box object
        sphere_b: Sphere object
        R_AB (batch, 3, 3): rotation from box to sphere model frames
        p_AoBo_A (batch, 3): vector from box to sphere in box frame

        Returns:
        phi (batch, n_c = 1): distance between objects
        R_AC (batch, n_c = 1, 3, 3): A model frame to contact frame [i.e. z == contact normal]
        p_AoAc_A (batch, n_c=1, 3): A's contact in A's frame
        p_BoBc_B (batch, n_c=1, 3): B's contact in B's frame
        """
        batch_dim = R_AB.shape[:-2]
        assert R_AB.shape == batch_dim + (3, 3)
        assert p_AoBo_A.shape == batch_dim + (3,)
        assert isinstance(box_a, Box)
        assert isinstance(sphere_b, Sphere)
        n_c = 2
        
        ## Get nearest point on box
        # Expand box lengths to batch size
        box_lengths = box_a.get_half_lengths().expand(p_AoBo_A.size())
        # Clamp to box
        p_AoBo_A_clamp = torch.clamp(p_AoBo_A, min=-box_lengths, max=box_lengths)
        # Project onto nearest face
        # Construct difference vector
        p_AoBo_A_clamp_sign = torch.sign(p_AoBo_A_clamp)
        p_AoBo_A_clamp_sign[p_AoBo_A_clamp_sign == 0.] = 1.
        p_AoBo_A_diffs = p_AoBo_A_clamp_sign*box_lengths - p_AoBo_A_clamp
        # Mask out all but the closest
        mask = torch.zeros_like(p_AoBo_A_diffs)
        mask[..., torch.arange(mask.shape[-2]), torch.argmin(torch.abs(p_AoBo_A_diffs), dim=1)] = 1.0
        p_AoBo_A_diffs_masked = p_AoBo_A_diffs * mask
        # Actual projection to get nearest point
        p_AoAc_A = p_AoBo_A_clamp + p_AoBo_A_diffs_masked

        # Get contact normal == normalized(nearest point -> center of the sphere)
        p_AcBo_A = p_AoBo_A - p_AoAc_A
        # Calculate directions (use torch nn functional normalize)
        directions_A = torch.nn.functional.normalize(p_AcBo_A, dim=-1)
        # Check if internal, if so, flip directions_A
        directions_A[torch.norm(p_AoBo_A, dim=1) < torch.norm(p_AoAc_A, dim=1)] *= -1.0
        # In the unlikely event p_AcBo_A == 0, use an arbitrary surface normal
        on_surface_idxs = (torch.norm(directions_A, dim=1) == 0)
        directions_A[on_surface_idxs] = -mask[on_surface_idxs]*p_AoBo_A_clamp_sign[on_surface_idxs]
        # Unsqueeze witness point dimensions
        directions_A = directions_A.unsqueeze(-2) 
        p_AoAc_A = p_AoAc_A.unsqueeze(-2)
        assert p_AoAc_A.shape == directions_A.shape == batch_dim + (1, 3) # (..., n_c == 1, 3)

        # Add estimated normal if they exist
        if estimated_normals_A is not None:
            assert estimated_normals_A.shape == batch_dim + (3,)
            directions_A2 = torch.nn.functional.normalize(estimated_normals_A, dim=-1)
            zeros_idx = torch.isclose(torch.norm(directions_A2, dim=-1), torch.zeros(batch_dim))
            directions_A2[zeros_idx, :] = directions_A[zeros_idx, 0, :]
            p_AoAc_A2 = box_a.support_points(directions_A2)[..., :1, :]
            p_AoAc_A = torch.cat([p_AoAc_A, p_AoAc_A2], dim=-2)
            directions_A = torch.cat([directions_A, directions_A2.unsqueeze(-2)], dim=-2)
        else:
            p_AoAc_A = p_AoAc_A.expand(batch_dim + (n_c, 3))
            directions_A = directions_A.expand(batch_dim + (n_c, 3))

        assert p_AoAc_A.shape == directions_A.shape == batch_dim + (n_c, 3) # (..., n_c == 2, 3)


        # directions needs to be (..., 1, 3) for pbmm, then re-squeezed
        directions_B = -pbmm(directions_A.unsqueeze(-2), R_AB.unsqueeze(-3)).squeeze(-2)

        # get support point of sphere
        # It adds n_c==1 which we can squeeze
        p_BoBc_B = sphere_b.support_points(directions_B).squeeze(-2)
        assert p_BoBc_B.shape == batch_dim + (n_c, 3) # (..., n_c == 2, 3)
        

        # Get R_AC by taking directions_a
        # Unsqueeze witness point dimension to 1
        R_AC = rotation_matrix_from_one_vector(directions_A, 2)
        assert R_AC.shape == batch_dim + (n_c, 3, 3) # (..., n_c == 2, 3, 3)

        # Get length of witness point distance projected onto contact normal
        p_BoBc_A = pbmm(p_BoBc_B.unsqueeze(-2), R_AB.unsqueeze(-3).expand(batch_dim + (n_c, 3, 3)).transpose(-1,-2)).squeeze(-2)
        p_AcBc_A = -p_AoAc_A + p_AoBo_A.unsqueeze(-2) + p_BoBc_A
        phi = torch.zeros(batch_dim + (n_c,))
        # Project Phi from Closest Point
        phi[..., :1] = (p_AcBc_A[..., :1, :] * R_AC[..., :1, :, 2]).sum(dim=-1)
        # Take vector norm of 2nd witness point
        phi[..., 1:] = torch.linalg.vector_norm(p_AcBc_A[..., 1:, :], dim=-1)
        #phi[..., 1:] = (p_AcBc_A[..., 1:, :] * R_AC[..., 1:, :, 2]).sum(dim=-1)
        assert phi.shape == batch_dim + (n_c,) # (..., n_c == 2)

        return phi, R_AC, p_AoAc_A, p_BoBc_B

    @staticmethod
    def collide_plane_convex(geometry_b: BoundedConvexCollisionGeometry,
                             R_AB: Tensor, p_AoBo_A: Tensor) -> \
            Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Implementation of ``GeometryCollider.collide()`` when
        ``geometry_a`` is a ``Plane`` and ``geometry_b`` is a
        ``BoundedConvexCollisionGeometry``."""
        R_BA = R_AB.transpose(-1, -2)

        # support direction on body B is negative z axis of A frame,
        # in B frame coordinates, i.e. the final column of ``R_BA``.
        directions_b = -R_BA[..., 2]

        # B support points of shape (*, N, 3)
        p_BoBc_B = geometry_b.support_points(directions_b)
        p_AoBc_A = pbmm(p_BoBc_B, R_BA) + p_AoBo_A.unsqueeze(-2)

        # phi is the A-axes z coordinate of Bc
        phi = p_AoBc_A[..., 2]

        # Ac is the projection of Bc onto the z=0 plane in frame A.
        # pylint: disable=E1103
        p_AoAc_A = torch.cat(
            (p_AoBc_A[..., :2], torch.zeros_like(p_AoBc_A[..., 2:])), -1)

        # ``R_AC`` (\*, N, 3, 3) is simply a batch of identities, as the z
        # axis of A points out of the plane.
        # pylint: disable=E1103
        R_AC = torch.eye(3).expand(p_AoAc_A.shape + (3,))
        return phi, R_AC, p_AoAc_A, p_BoBc_B

    @staticmethod
    def collide_convex_convex(
            geometry_a: BoundedConvexCollisionGeometry, geometry_b: BoundedConvexCollisionGeometry,
            R_AB: Tensor,
            p_AoBo_A: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Implementation of ``GeometryCollider.collide()`` when
        both geometries are ``BoundedConvexCollisionGeometry``\es."""

        # Call network directly for DeepSupportConvex objects
        support_fn_a = geometry_a.support_points
        support_fn_b = geometry_b.support_points
        if isinstance(geometry_a, DeepSupportConvex):
            support_fn_a = geometry_a.network
        if isinstance(geometry_b, DeepSupportConvex):
            support_fn_b = geometry_b.network

        # pylint: disable=too-many-locals
        p_AoBo_A = p_AoBo_A.unsqueeze(-2)
        original_batch_dims = p_AoBo_A.shape[:-2]
        p_AoBo_A = p_AoBo_A.view(-1, 3)
        R_AB = R_AB.view(-1, 3, 3)
        batch_range = p_AoBo_A.shape[0]

        # Assume collision directions are piecewise constant, which allows us
        # to use :py:mod:`fcl` to compute the direction, without the need to
        # differentiate through it.
        # pylint: disable=E1103
        directions = torch.zeros_like(p_AoBo_A)

        # Used if there are multiple coplanar witness points
        # To determine the actual contact point in a differentiable manner
        hints_a = torch.zeros_like(p_AoBo_A)
        hints_b = torch.zeros_like(p_AoBo_A)

        # setup fcl=
        a_obj = fcl.CollisionObject(geometry_a.get_fcl_geometry(),
                                    fcl.Transform())
        b_obj = fcl.CollisionObject(geometry_b.get_fcl_geometry(),
                                    fcl.Transform())
        collision_request = fcl.CollisionRequest()
        collision_request.enable_contact = True
        distance_request = fcl.DistanceRequest()
        distance_request.enable_nearest_points = True

        for transform_index in range(batch_range):
            b_t = fcl.Transform(R_AB[transform_index].detach().cpu().numpy(),
                                p_AoBo_A[transform_index].detach().cpu().numpy())
            b_obj.setTransform(b_t)
            result = fcl.CollisionResult()
            if fcl.collide(a_obj, b_obj, collision_request, result) > 0:
                # Collision detected.
                # Assume only 1 contact point.
                directions[transform_index] += torch.tensor(result.contacts[0].normal)
                nearest_points = [result.contacts[0].pos + result.contacts[0].penetration_depth/2.0 * result.contacts[0].normal,
                    result.contacts[0].pos - result.contacts[0].penetration_depth/2.0 * result.contacts[0].normal]
            else:
                result = fcl.DistanceResult()
                fcl.distance(a_obj, b_obj, distance_request, result)
                directions[transform_index] += torch.tensor(result.nearest_points[1] -
                                                      result.nearest_points[0])

                nearest_points = result.nearest_points

            # Record Hints == expected contact point in each object's frame
            hints_a[transform_index] = torch.tensor(nearest_points[0])
            hints_b[transform_index] = pbmm(
                torch.tensor(nearest_points[1]) - p_AoBo_A[transform_index].detach(), 
                R_AB[transform_index])

        # Get normal directions in each object frame
        directions_A = directions / directions.norm(dim=-1, keepdim=True)
        directions_B = -pbmm(directions_A.unsqueeze(-2), R_AB).squeeze(-2)

        p_AoAc_A = support_fn_a(directions_A, hints_a)
        p_BoBc_B = support_fn_b(directions_B, hints_b)
        p_BoBc_A = pbmm(p_BoBc_B, R_AB.transpose(-1,-2))
        # Check Sanity of autodiff-calculated points relative to FCL
        assert np.isclose(p_AoAc_A.detach().cpu().numpy(), hints_a.unsqueeze(-2).detach().cpu().numpy()).all()
        assert np.isclose(p_BoBc_B.detach().cpu().numpy(), hints_b.unsqueeze(-2).detach().cpu().numpy()).all()

        p_AcBc_A = -p_AoAc_A + p_AoBo_A.unsqueeze(-2) + p_BoBc_A

        # Unsqueeze # of witness points dimension, default 1
        R_AC = rotation_matrix_from_one_vector(directions_A, 2).unsqueeze(-3)
        # Assume same contact frame for all witness points
        R_AC = R_AC.expand(p_AcBc_A.shape + (3,))
        # Get length of witness point distance projected onto contact normal
        phi = (p_AcBc_A * R_AC[..., 2]).sum(dim=-1)       
        
        # No longer necessary
        # phi = phi.reshape(original_batch_dims + (1,))
        # R_AC = R_AC.reshape(original_batch_dims + (1, 3, 3))
        # p_AoAc_A = p_AoAc_A.reshape(original_batch_dims + (1, 3))
        # p_BoBc_B = p_BoBc_B.reshape(original_batch_dims + (1, 3))

        # Check outputs are sane before return
        assert (phi.shape + (3,3,)) == R_AC.shape
        assert phi.shape[1] == 1 # TODO: HACK Only supporting 1 contact witness point
        assert phi.shape[1] == p_AoAc_A.shape[1]
        assert phi.shape[1] == p_BoBc_B.shape[1]
        return phi, R_AC, p_AoAc_A, p_BoBc_B
