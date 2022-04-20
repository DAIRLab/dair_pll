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
from abc import ABC, abstractmethod
from typing import Tuple, Dict

import fcl
import numpy as np
import pywavefront  # type: ignore
import torch
from pydrake.geometry import Box as DrakeBox  # type: ignore
from pydrake.geometry import HalfSpace as DrakeHalfSpace  # type: ignore
from pydrake.geometry import Mesh as DrakeMesh  # type: ignore
from pydrake.geometry import Shape  # type: ignore
from torch import Tensor
from torch.nn import Module, Parameter

from dair_pll.deep_support_function import HomogeneousICNN, \
    extract_outward_normal_hyperplanes, extract_mesh
from dair_pll.tensor_utils import pbmm, tile_dim, \
    rotation_matrix_from_one_vector

_UNIT_BOX_VERTICES = Tensor([[0, 0, 0, 0, 1, 1, 1, 1.], [
    0, 0, 1, 1, 0, 0, 1, 1.
], [0, 1, 0, 1, 0, 1, 0, 1.]]).t() * 2. - 1.

_ROT_Z_45 = Tensor([[2**(-0.5), -(2**(-0.5)), 0.], [2**(-0.5), 2**(-0.5), 0.],
                    [0., 0., 1.]])

_total_ordering = ['Plane', 'Polygon', 'Box', 'Sphere', 'DeepSupportConvex']

_POLYGON_DEFAULT_N_QUERY = 4
_DEEP_SUPPORT_DEFAULT_N_QUERY = 4
_DEEP_SUPPORT_DEFAULT_DEPTH = 2
_DEEP_SUPPOR_DEFAULTT_WIDTH = 256


class CollisionGeometry(ABC, Module):
    """Abstract interface for collision geometries.

    Collision geometries have heterogeneous implementation depending on the
    type of shape. This class mainly enforces the implementation of
    bookkeeping interfaces.

    When two collision geometries are evaluated for collision with
    ``GeometryCollider``, their ordering is constrained via a total order on
    collision geometry types, enforced with an overload of the ``>`` operator.
    """

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
    def support_points(self, directions: Tensor) -> Tensor:
        """Returns a set of witness points representing contact with another
        shape off in the direction(s) ``directions``.

        This method will return a set of points ``S' \\subset S`` such that::

            argmax_{s \\in S} s \\cdot directions \\subset convexHull(S').


        In theory, returning exactly the argmax set would be sufficient.
        However,

        Args:
            directions: (*, 3) batch of unit-length directions.

        Returns:
            (*, N, 3) sets of corresponding witness points of cardinality N.
        """


class SparseVertexConvexCollisionGeometry(BoundedConvexCollisionGeometry):
    """Partial implementation of ``BoundedConvexCollisionGeometry`` when
    witness points are guaranteed to be contained in a small set of vertices.

    An obvious subtype is any sort of Polytope, such as a ``Box``. A less
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

    def support_points(self, directions: Tensor) -> Tensor:
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
            directions: (*, 3) batch of directions.

        Returns:
            (*, n_query, 3) sets of corresponding witness points.
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
                                sorted=False).indices.t()

        top_vertices = torch.stack(
            [vertices[batch_range, selection] for selection in selections], -2)
        # reshape to (*, n_query, 3)
        return top_vertices.view(original_shape[:-1] + (self.n_query, 3))

    @abstractmethod
    def get_vertices(self, directions: Tensor) -> Tensor:
        """Returns sparse witness point set as collection of vertices.

        Specifically, given search directions, returns a set of points
        ``S_v`` for which::

            argmax_{s \\in S} s \\cdot directions \\subset convexHull(S_v).

        Args:
            directions: (*, 3) batch of unit-length directions.
        Returns:
            (*, N, 3) witness point convex hull vertices.
        """


class Polygon(SparseVertexConvexCollisionGeometry):
    """Concrete implementation of a convex polytope.

    Implemented via ``SparseVertexConvexCollisionGeometry`` as a static set
    of vertices, where models the underlying shape as all convex combinations
    of the vertices.
    """
    vertices: Parameter

    def __init__(self,
                 vertices: Tensor,
                 n_query: int = _POLYGON_DEFAULT_N_QUERY) -> None:
        """Inits ``Polygon`` object with initial vertex set.

        Args:
            vertices: (N, 3) static vertex set.
            n_query: number of vertices to return in witness point set.
        """
        super().__init__(n_query)
        self.vertices = Parameter(vertices.clone(), requires_grad=True)

    def get_vertices(self, directions: Tensor) -> Tensor:
        """Return batched view of static vertex set"""
        return self.vertices.expand(directions.shape[:-1] + self.vertices.shape)

    def scalars(self) -> Dict[str, float]:
        """Return one scalar for each vertex index."""
        scalars = {}
        axes = ['x', 'y', 'z']
        for axis, values in zip(axes, self.vertices.t()):
            for vertex_index, value in enumerate(values):
                scalars[f'v{vertex_index}_{axis}'] = value.item()
        return scalars


class DeepSupportConvex(SparseVertexConvexCollisionGeometry):
    r"""Deep support function convex shape.

    Any convex shape :math:`S` can be equivalently represented via its support
    function :math:`f(d)`\ , which returns the extent to which the object
    extends in the :math:`d` direction:

    .. math::

        f(d) = \max_{s \in S} s \cdot d.

    Given a direction, the set of points that form the :math:`\arg\max` in
    :math:`f(d)` is exactly the convex subgradient :math:`\partial_d f(d)`\ .

    Furthermore, for every convex shape, :math:`f(d)` is convex and
    positively homogeneous, and every convex and positively homogeneous
    :math:`f(d)` is the support function of some convex shape.

    This collision geometry type implements the support function directly as
    a convex and positively homogeneous neural network (
    :py:class:`~dair_pll.deep_support_function.HomogeneousICNN`\ )."""
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
                 width: int = _DEEP_SUPPOR_DEFAULTT_WIDTH,
                 perturbation: float = 0.1) -> None:
        """Inits ``DeepSupportConvex`` object with initial vertex set.

        When calculating a sparse vertex set with :py:meth:`get_vertices`\ ,
        supplements the support direction with nearby directions randomly.

        Args:
            vertices: ``(N, 3)`` initial vertex set.
            n_query: Number of vertices to return in witness point set.
            depth: Depth of support function network.
            width: Width of support function network.
            perturbation: support direction sampling parameter.
        """
        super().__init__(n_query)
        length_scale = (vertices.max(dim=0).values -
                        vertices.min(dim=0).values).norm() / 2
        self.network = HomogeneousICNN(depth, width, scale=length_scale)
        self.perturbations = torch.cat((torch.zeros(
            (1, 3)), perturbation * (torch.rand((n_query - 1, 3)) - 0.5)))

    def get_vertices(self, directions: Tensor) -> Tensor:
        """Return batched view of support points of interest.

        Given a direction :batch:`d`, this function finds the support point
        of the object in that direction, calculated via envelope

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

    def train(self, mode: bool = True) -> Module:
        """Override training-mode setter from :py:mod:`torch`\ .

        Sets a static fcl mesh geometry for the entirety of evaluation time,
        as the underlying support function is not changing.

        Args:
            mode: ``True`` for training, ``False`` for evaluation.

        Returns:
            ``self``\ .
        """
        if not mode:
            self.fcl_geometry = self.get_fcl_geometry()
        return super().train(mode)

    def get_fcl_geometry(self) -> fcl.BVHModel:
        """Retrieves :py:mod:`fcl` mesh collision geometry representation.

        If evaluation mode is set, retrieves precalculated version.

        Returns:
            :py:mod:`fcl`bounding volume hierarchy for mesh.
        """
        if self.training:
            mesh = extract_mesh(self.network)
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
    """Implementation of cuboid geometry as a sparse vertex convex hull."""
    half_lengths: Parameter
    unit_vertices: Tensor

    def __init__(self, half_lengths: Tensor, n_query: int) -> None:
        """Inits ``Box`` object with initial size.

        Args:
            half_lengths: (3,) half-length dimensions of box on x, y,
              and z axes.
            n_query: number of vertices to return in witness point set.
        """
        super().__init__(n_query)

        assert half_lengths.numel() == 3

        self.half_lengths = Parameter(half_lengths.clone().view(1, -1),
                                      requires_grad=True)
        self.unit_vertices = _UNIT_BOX_VERTICES.clone()

    def get_vertices(self, directions: Tensor) -> Tensor:
        """Returns view of cuboid's static vertex set."""
        return (self.unit_vertices *
                self.half_lengths).expand(directions.shape[:-1] +
                                          self.unit_vertices.shape)

    def scalars(self) -> Dict[str, float]:
        """Returns each axis's full length as a scalar."""
        scalars = {
            f'len_{axis}': 2 * value.item()
            for axis, value in zip(['x', 'y', 'z'], self.half_lengths.view(-1))
        }
        return scalars


class Sphere(BoundedConvexCollisionGeometry):
    """Implements sphere geometry via its support function.

    It is trivial to calculate the witness point for a sphere contact as
    simply the product of the sphere's radius and the support direction.
    """

    def __init__(self, radius: Tensor) -> None:
        super().__init__()
        assert radius.numel == 1

        self.radius = Parameter(radius.clone().view(()), requires_grad=True)

    def support_points(self, directions: Tensor) -> Tensor:
        """Implements ``BoundedConvexCollisionGeometry.support_points()``
        via analytic expression::

            argmax_{s \\in S} s \\cdot directions = directions * radius.

        Args:
            directions: (*, 3) batch of directions.

        Returns:
            (*, 1, 3) corresponding witness point sets of cardinality 1.
        """
        return (directions.clone() * self.radius).unsqueeze(-2)

    def scalars(self) -> Dict[str, float]:
        """Logs radius as a scalar."""
        return {'radius': self.radius.item()}


class PydrakeToCollisionGeometryFactory:
    """Utility class for converting Drake ``Shape`` instances to
    ``CollisionGeometry`` instances."""

    @staticmethod
    def convert(drake_shape: Shape) -> CollisionGeometry:
        """Converts abstract ``pydrake.geometry.shape`` to
        ``CollisionGeometry``.

        Args:
            drake_shape: drake shape type to convert.

        Returns:
            Collision geometry representation of shape.

        Raises:
            TypeError: When provided object is not a supported Drake shape type.
        """
        if isinstance(drake_shape, DrakeBox):
            return PydrakeToCollisionGeometryFactory.convert_box(drake_shape)
        if isinstance(drake_shape, DrakeHalfSpace):
            return PydrakeToCollisionGeometryFactory.convert_plane()
        if isinstance(drake_shape, DrakeMesh):
            return PydrakeToCollisionGeometryFactory.convert_mesh(drake_shape)
        raise TypeError(
            "Unsupported type for drake Shape() to"
            "CollisionGeometry() conversion:", type(drake_shape))

    @staticmethod
    def convert_box(drake_box: DrakeBox) -> Box:
        """Converts ``pydrake.geometry.Box`` to ``Box``"""
        half_widths = 0.5 * Tensor(np.copy(drake_box.size()))
        return Box(half_widths, 4)

    @staticmethod
    def convert_plane() -> Plane:
        """Converts ``pydrake.geometry.HalfSpace`` to ``Plane``"""
        return Plane()

    @staticmethod
    def convert_mesh(drake_mesh: DrakeMesh) -> DeepSupportConvex:
        """Converts ``pydrake.geometry.Mesh`` to ``Polygon``"""
        filename = drake_mesh.filename()
        mesh = pywavefront.Wavefront(filename)
        vertices = Tensor(mesh.vertices)
        return DeepSupportConvex(vertices)

class GeometryCollider:
    """Utility class for colliding two ``CollisionGeometry`` instances."""

    @staticmethod
    def collide(geometry_a: CollisionGeometry, geometry_b: CollisionGeometry,
                R_AB: Tensor, p_AoBo_A: Tensor) -> \
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
            R_AB: (*,3,3) rotation between geometry frames
            p_AoBo_A: (*, 3) offset of geometry frame origins

        Returns:
            (*, N) batch of witness point pair distances
            (*, N, 3, 3) contact frame C rotation in A, R_AC, where the z
            axis of C is contained in the normal cone of body A at contact
            point Ac and is parallel (or antiparallel) to AcBc.
            (*, N, 3) witness points Ac on A, p_AoAc_A
            (*, N, 3) witness points Bc on B, p_BoBc_B
        """
        assert not geometry_a > geometry_b

        # case 1: half-space to compact-convex collision
        if isinstance(geometry_a, Plane) and isinstance(
                geometry_b, BoundedConvexCollisionGeometry):
            return GeometryCollider.collide_plane_convex(
                geometry_b, R_AB, p_AoBo_A)
        if isinstance(geometry_a, DeepSupportConvex) and isinstance(
                geometry_b, DeepSupportConvex):
            return GeometryCollider.collide_mesh_mesh(geometry_a,
                                                      geometry_b, R_AB, p_AoBo_A)
        raise TypeError(
            "No type-specific implementation for geometry "
            "pair of following types:",
            type(geometry_a).__name__,
            type(geometry_b).__name__)

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

        # ``R_AC`` (*, N, 3, 3) is simply a batch of identities, as the z
        # axis of A points out of the plane.
        # pylint: disable=E1103
        R_AC = torch.eye(3).expand(p_AoAc_A.shape + (3,))
        return phi, R_AC, p_AoAc_A, p_BoBc_B

    @staticmethod
    def collide_gjk(geometry_a: DeepSupportConvex,
                    geometry_b: DeepSupportConvex,
                    R_AB: Tensor,
                    p_AoBo_A: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        original_batch = p_AoBo_A.shape[:-1]
        p_AoBo_A = p_AoBo_A.view(-1, 3)
        R_AB = R_AB.view(-1, 3, 3)
        batch_size = p_AoBo_A.shape[0]
        batch_range = torch.arange(batch_size)
        def supports(a_directions, R_ab, p_ab_a):
            points_a = geometry_a.network(a_directions).detach()
            b_directions = pbmm(-a_directions.unsqueeze(-2), R_ab).squeeze(-2)
            points_b = geometry_b.network(b_directions).detach()
            return points_a - points_b - p_ab_a

        def same_direction(dir_a, dir_b, eps=0.0):
            dots = (dir_a * dir_b).sum(dim=-1)
            return dots > eps

        def line_update(lines, ranks):
            ranks[:] = 2
            p_a = lines[:, 1, :]
            p_b = lines[:, 0, :]
            p_ab = p_b - p_a
            p_ao = -p_a
            directions = p_ao
            same = same_direction(p_ab, p_ao)
            directions[same] = p_ab[same].cross(p_ao[same]).cross(p_ab[same])
            lines[~same, 0, :] = p_a[~same]
            ranks[~same] = 1
            collisions = torch.ones_like(ranks) < 0
            return lines, ranks, directions, collisions

        def triangle_update(triangles, ranks):
            ranks[:] = 3
            p_a = triangles[:, 2, :]
            p_b = triangles[:, 1, :]
            p_c = triangles[:, 0, :]
            p_ab = p_b - p_a
            p_ac = p_c - p_a
            p_ao = -p_a
            p_abc = torch.cross(p_ab, p_ac)
            check_1 = same_direction(p_abc.cross(p_ac), p_ao)
            check_2 = same_direction(p_ac, p_ao)
            check_3 = same_direction(p_ab.cross(p_abc), p_ao)
            check_4 = same_direction(p_abc, p_ao)
            case_1 = check_1 & check_2
            case_2 = (check_1 & ~check_2) | (~check_1 & check_3)
            case_3 = ~check_1 & (check_3 & check_4)
            case_4 = ~check_1 & (check_3 & ~check_4)
            directions = p_abc.clone()
            triangles[case_1, :2] = torch.stack((
                p_c[case_1], p_a[case_1]), dim=-2)
            ranks[case_1] = 2
            directions[case_1] = p_ac[case_1].cross(p_ao[case_1]).cross(
                p_ac[case_1])
            triangles[case_2, :2] = triangles[case_2, 1:3]
            triangles[case_2], ranks[case_2], directions[case_2], _ \
                = line_update(triangles[case_2], ranks[case_2])
            # case_3 is default
            directions[case_4] *= -1
            collisions = torch.ones_like(ranks) < 0
            return triangles, ranks, directions, collisions

        def tet_update(tets, ranks):
            ranks[:] = 4
            p_a = tets[:, 3, :]
            p_b = tets[:, 2, :]
            p_c = tets[:, 1, :]
            p_d = tets[:, 0, :]

            p_ab = p_b - p_a
            p_ac = p_c - p_a
            p_ad = p_d - p_a
            p_ao = -p_a

            case_1 = same_direction(p_ab.cross(p_ac), p_ao)
            case_2 = same_direction(p_ac.cross(p_ad), p_ao)
            case_3 = same_direction(p_ad.cross(p_ab), p_ao)
            case_4 = ~(case_1 | case_2 | case_3)

            tets[case_1, :3] = torch.stack(
                (p_c[case_1], p_b[case_1], p_a[case_1]), dim=-2)
            tets[case_2, :3] = torch.stack(
                (p_d[case_2], p_c[case_2], p_a[case_2]), dim=-2)
            tets[case_3, :3] = torch.stack(
                (p_b[case_3], p_d[case_3], p_a[case_3]), dim=-2)
            directions = torch.zeros_like(p_a)
            for case in [case_1, case_2, case_3]:
                tets[case], ranks[case], directions[case], _ = triangle_update(
                    tets[case], ranks[case]
                )

            return tets, ranks, directions, case_4

        def simplex_update(simplices, ranks):
            lines = ranks == 2
            triangles = ranks == 3
            tets = ranks == 4
            collision = torch.ones_like(ranks) < 0
            directions = torch.zeros_like(simplices[:,0,:])
            simplices[lines], ranks[lines], directions[lines], _ = line_update(
                simplices[lines], ranks[lines])
            simplices[triangles], ranks[triangles], directions[triangles], _ \
                = triangle_update(simplices[triangles], ranks[triangles])
            simplices[tets], ranks[tets], directions[tets], collision[tets] = \
                tet_update(simplices[tets], ranks[tets])
            directions /= directions.norm(dim= -1, keepdim=True)
            return simplices, ranks, directions[~collision], collision

        def get_face_normals(polytopes, faces):
            all_normals, _, all_distances = \
                extract_outward_normal_hyperplanes(polytopes, faces)
            min_distances = all_distances.min(dim=-1)
            return all_normals, min_distances.indices, min_distances.values



        init_directions = p_AoBo_A / p_AoBo_A.norm(dim=-1, keepdim=True)
        init_supports = supports(init_directions, R_AB, p_AoBo_A)
        simplices = init_supports.unsqueeze(-2).repeat((1,4,1))
        ranks = torch.ones((batch_size,)).to(torch.long)
        has_collision = ranks.clone() == 0  # default to all false
        cur_directions = -init_supports / init_supports.norm(
            dim=-1, keepdim=True)
        cur = torch.arange(batch_size)
        for _ in range(100):
            N_cont = cur.shape[0]
            print(_,N_cont)
            if N_cont == 0:
                break
            new_points = supports(cur_directions, R_AB[cur], p_AoBo_A[cur])
            continues = same_direction(cur_directions, new_points, eps=1e-7)
            # freeze rejections; result is no collision by default
            cur = cur[continues]
            new_points = new_points[continues]

            # push to front
            assert cur.shape[0] == new_points.shape[0]
            simplices[cur,ranks[cur]] = new_points
            ranks[cur] += 1

            # update simplices
            simplices[cur], ranks[cur], cur_directions, collisions = \
                simplex_update(simplices[cur], ranks[cur])
            # freeze collisions
            collides = cur[collisions]
            has_collision[collides] = True
            cur = cur[~collisions]
            assert cur.shape[0] == cur_directions.shape[0]
        print(has_collision.sum()/batch_size)
        """
        # should all be tets
        cur_polytopes = simplices[has_collision]
        faces = Tensor([[0, 1, 2],
		                [0, 3, 1],
		                [0, 2, 3],
		                [1, 3, 2]]).to(torch.long)
        faces = faces.repeat((cur_polytopes.shape[0],1,1))
        cur_faces = faces
        cur_normals, min_faces, min_distances = get_face_normals(
            cur_polytopes, cur_faces)
        final_normals = cur_normals.clone()
        final_distances = min_distances.clone()
        cur = torch.arange(batch_size)
        poly_size = 4
        EPS = 1e-3
        while cur.shape[0] > 0:
            cur_range = torch.arange(cur.shape[0])
            min_normals = cur_normals[cur_range, min_faces]
            support = supports(min_normals, R_AB[cur], p_AoBo_A[cur])
            s_distances = (support * min_normals).sum(dim=-1)
            needs_expansion = (s_distances - min_distances).abs() > EPS

            cur = cur[needs_expansion]
            cur_polytopes = cur_polytopes[needs_expansion]
            cur_normals = cur_normals[needs_expansion]
            cur_faces = cur_faces[needs_expansion]
            cur_range = torch.arange(cur.shape[0])

            support_facing_normals = same_direction(cur_normals,
                                                    support.unsqueeze(-2))
            single_valid_face = support_facing_normals.max(dim=-1).indices
            removed_count = support_facing_normals.sum(dim=-1).max()
            sfn_transformed = support_facing_normals.sort(dim=-1,
                descending=True).values[:,:removed_count]
            removed_faces = single_valid_face.unsqueeze(-1).repeat((1,removed_count))
            removed_faces[sfn_transformed] = faces[support_facing_normals]
            edges = []
            for (i,j) in [(0,1), (1,2), (2,0)]:
                edges.append(torch.stack((removed_faces[..., i],
                                          removed_faces[..., j])), dim=-1)
            edges = torch.stack(edges, dim=-1)
        """

    @staticmethod
    def collide_mesh_mesh(geometry_a: DeepSupportConvex,
                          geometry_b: DeepSupportConvex,
                          R_AB: Tensor,
                          p_AoBo_A: Tensor) -> Tuple[Tensor, Tensor, Tensor,
                                                     Tensor]:
        """Implementation of ``GeometryCollider.collide()`` when
         both geometries are ``DeepSupportConvex``\ es."""

        p_AoBo_A = p_AoBo_A.unsqueeze(-2)
        original_batch_dims = p_AoBo_A.shape[:-2]
        p_AoBo_A = p_AoBo_A.view(-1, 3)
        R_AB = R_AB.view(-1, 3, 3)
        batch_range = p_AoBo_A.shape[0]

        # Assume collision directions are piecewise constant, which allows us
        # to use :py:mod:`fcl` to compute the direction, without the need to
        # differentiate through it.
        directions = torch.zeros_like(p_AoBo_A)

        # setup fcl
        a_fcl = geometry_a.get_fcl_geometry()
        b_fcl = geometry_b.get_fcl_geometry()
        a_obj = fcl.CollisionObject(a_fcl, fcl.Transform())
        b_obj = fcl.CollisionObject(b_fcl, fcl.Transform())
        collision_request = fcl.CollisionRequest()
        collision_request.enable_contact = True
        distance_request = fcl.DistanceRequest()
        distance_request.enable_nearest_points = True

        for i in range(batch_range):
            b_t = fcl.Transform(R_AB[i].detach().numpy(),
                                p_AoBo_A[i].detach().numpy())
            b_obj.setTransform(b_t)
            result = fcl.CollisionResult()
            has_collided = fcl.collide(a_obj, b_obj, collision_request,
                                       result) > 0
            if has_collided:
                # Assume only 1 contact point
                directions[i] += result.contacts[0].normal
            else:
                result = fcl.DistanceResult()
                fcl.distance(a_obj, b_obj, distance_request, result)
                dp = Tensor(result.nearest_points[1] - result.nearest_points[0])
                directions[i] += dp
        directions /= directions.norm(dim=-1, keepdim=True)
        R_AC = rotation_matrix_from_one_vector(directions, 2)
        p_AoAc_A = geometry_a.network(directions)
        b_dirs = -pbmm(directions.unsqueeze(-2), R_AB).squeeze(-2)
        p_BoBc_B = geometry_b.network(b_dirs)
        p_BoBc_A = pbmm(p_BoBc_B.unsqueeze(-2),
                        R_AB.transpose(-1,-2)).squeeze(-2)
        p_AcBc_A = -p_AoAc_A + p_AoBo_A + p_BoBc_A
        phi = (p_AcBc_A * R_AC[...,2]).sum(dim=-1)
        return phi.reshape(original_batch_dims + (1,)), \
               R_AC.reshape(original_batch_dims + (1,3,3)), \
               p_AoAc_A.reshape(original_batch_dims + (1,3)), \
               p_BoBc_B.reshape(original_batch_dims + (1,3))


if __name__ == "__main__":
    LS = 0.5
    B = 64
    R = 64
    TR = 256
    TESTS = 2 ** 14
    PA = (torch.rand((TR,3)) - 0.5) * 2 * LS
    PB = (torch.rand((TR,3)) - 0.5) * 2 * LS
    geometry_a = DeepSupportConvex(PA, width=256)
    geometry_b = DeepSupportConvex(PB, width=256)
    tests = (torch.rand((TESTS,3)) - 0.5)
    tests /= tests.norm(dim=-1,keepdim=True)
    supps_a = geometry_a.network.network_activations(tests)[1]
    supps_b = geometry_b.network.network_activations(tests)[1]
    min_to_gap = (supps_a.min() + supps_b.min()).detach()
    max_to_gap = (supps_a.max() + supps_b.max()).detach()
    print(min_to_gap, max_to_gap)
    p_AoBo_A = torch.rand((B,3)) - 0.5
    p_AoBo_A *= min_to_gap * 0.999 / p_AoBo_A.norm(dim=-1,keepdim=True)
    #p_AoBo_A *= max_to_gap * 1.001 / p_AoBo_A.norm(dim=-1,keepdim=True)
    xhat = torch.rand((B,3))
    xhat /= xhat.norm(dim=-1, keepdim=True)
    #R_AB = rotation_matrix_from_one_vector(xhat, axis=0)
    R_AB = torch.eye(3).unsqueeze(0).repeat((B,1,1))
    #GeometryCollider.collide_gjk(geometry_a,geometry_b, R_AB, p_AoBo_A)
    import time
    print('start')
    st = time.time()
    for i in range(R):
        GeometryCollider.collide_mesh_mesh(geometry_a, geometry_b, R_AB, p_AoBo_A)
    print((time.time() - st)/(B*R))
    print('end')
