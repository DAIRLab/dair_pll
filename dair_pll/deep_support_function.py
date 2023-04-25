"""Modelling and manipulation of convex support functions."""
from typing import Callable, Tuple, List

import pdb

import torch
import torch.nn
from scipy.spatial import ConvexHull  # type: ignore
from torch import Tensor
from torch.nn import Parameter, ParameterList, Module

from dair_pll.system import MeshSummary
from dair_pll.tensor_utils import pbmm, rotation_matrix_from_one_vector

# pylint: disable=E1103
_LINEAR_SPACE = torch.linspace(-1, 1, steps=8)
_GRID = torch.cartesian_prod(_LINEAR_SPACE, _LINEAR_SPACE, _LINEAR_SPACE)
_SURFACE = _GRID[_GRID.abs().max(dim=-1).values >= 1.0]
_SURFACE = _SURFACE / _SURFACE.norm(dim=-1, keepdim=True)
_SURFACE_ROTATIONS = rotation_matrix_from_one_vector(_SURFACE, 2)

_POLYGON_DEFAULT_FACES = torch.tensor([[0, 1, 2], [2, 1, 3], [2, 3, 4],
                                       [4, 3, 5], [4, 5, 6], [6, 5, 7],
                                       [6, 7, 0], [0, 7, 1], [1, 7, 3],
                                       [3, 7, 5], [6, 0, 4], [4, 0, 2]],
                                      dtype=torch.int64)


def get_mesh_summary_from_polygon(polygon) -> MeshSummary:
    """Assuming a standard ordering of vertices for a ``Polygon``
    representation, produce a ``MeshSummary`` of this sparse mesh.

    Note:
        This is a hack since it only works for ``Polygon``\s of a particular
        structure.  That structure matches that provided in the example assets
        ``contactnets_cube.obj`` and ``contactnets_elbow_half.obj``. 

    Args:
        polygon: A ``Polygon`` ``CollisionGeometry``.

    Returns:
        A ``MeshSummary`` of the polygon.
    """
    return MeshSummary(vertices=polygon.vertices, faces=_POLYGON_DEFAULT_FACES)


def extract_obj_from_support_function(
    support_function: Callable[[Tensor], Tensor]) -> str:
    """Given a support function, extracts a Wavefront obj representation.

    Args:
        support_function: Callable support function.

    Returns:
        Wavefront .obj string
    """
    mesh_summary = extract_mesh_from_support_function(support_function)
    return extract_obj_from_mesh_summary(mesh_summary)


def extract_obj_from_mesh_summary(mesh_summary: MeshSummary) -> str:
    """Given a mesh summary, extracts a Wavefront obj representation.

    Args:
        mesh_summary: Object vertices and face indices in the form of a
          ``MeshSummary``.

    Returns:
        Wavefront .obj string
    """
    normals = extract_outward_normal_hyperplanes(
        mesh_summary.vertices.unsqueeze(0),
        mesh_summary.faces.unsqueeze(0)
    )[0].squeeze(0)

    obj_string = ""
    for vertex in mesh_summary.vertices:
        vertex_string = " ".join([str(v_i.item()) for v_i in vertex])
        obj_string += f'v {vertex_string}\n'

    obj_string += '\n\n'

    for normal in normals:
        normal_string = " ".join([str(n_i.item()) for n_i in normal])
        obj_string += f'vn {normal_string}\n'

    obj_string += '\n\n'

    for face_index, face in enumerate(mesh_summary.faces):
        face_string = " ".join([f'{f_i.item() + 1}//{face_index + 1}' for f_i in face])
        obj_string += f'f {face_string}\n'

    return obj_string


def extract_outward_normal_hyperplanes(vertices: Tensor, faces: Tensor):
    r"""Extract hyperplane representation of convex hull from vertex-plane
    representation.

    Constructs a set of (outward) normal vectors and intercept values.
    Additionally, notes a boolean value that is ``True`` iff the face vertices
    are in counter-clockwise order when viewed from the outside.

    Mathematically for a face :math:`(v_1, v_2, v_3)`\ , in counter-clockwise
    order, this function returns :math:`\hat n`\ , the unit vector in the
    :math:`(v_2 - v_1) \times (v_3 - v_`1)` direction, and intercept
    :math:`d = \hat n \cdot v_1`\ .

    Args:
        vertices: ``(*, N, 3)`` batch of polytope vertices.
        faces: ``(*, M, 3)`` batch of polytope triangle face vertex indices.

    Returns:
        ``(*, M, 3)`` face outward normals.
        ``(*, M)`` whether each face is in counter-clockwise order.
        ``(*, M)`` face hyperplane intercepts.
    """
    batch_range = torch.arange(vertices.shape[0]).unsqueeze(1).repeat(
        (1, faces.shape[-2]))
    centroids = vertices.mean(dim=-2, keepdim=True)
    v_a = vertices[batch_range, faces[..., 0]]
    v_b = vertices[batch_range, faces[..., 1]]
    v_c = vertices[batch_range, faces[..., 2]]
    outward_normals = torch.cross(v_b - v_a, v_c - v_a)
    outward_normals /= outward_normals.norm(dim=-1, keepdim=True)
    backwards = (outward_normals * (v_a - centroids)).sum(dim=-1) < 0.
    outward_normals[backwards] *= -1
    extents = (v_a * outward_normals).sum(dim=-1)
    return outward_normals, backwards, extents


def extract_mesh_from_support_function(
    support_function: Callable[[Tensor], Tensor]) -> MeshSummary:
    """Given a support function, extracts a vertex/face mesh.

    Args:
        support_function: Callable support function.

    Returns:
        Object vertices and face indices.
    """
    support_points = support_function(_SURFACE).detach()
    support_point_hashes = set()
    unique_support_points = []

    # remove duplicate vertices
    for vertex in support_points:
        vertex_hash = hash(vertex.numpy().tobytes())
        if vertex_hash in support_point_hashes:
            continue
        support_point_hashes.add(vertex_hash)
        unique_support_points.append(vertex)

    vertices = torch.stack(unique_support_points)
    hull = ConvexHull(vertices.numpy())
    faces = Tensor(hull.simplices).to(torch.long)  # type: ignore

    _, backwards, _ = extract_outward_normal_hyperplanes(
        vertices.unsqueeze(0), faces.unsqueeze(0))
    backwards = backwards.squeeze(0)
    faces[backwards] = faces[backwards].flip(-1)

    return MeshSummary(vertices=support_points, faces=faces)


class HomogeneousICNN(Module):
    r""" Homogeneous Input-convex Neural Networks.

    Implements a positively-homogenous version of an ICNN :cite:p:`AmosICNN2017`\ .

    These networks have the structure :math:`f(d)` where

    .. math::
        \begin{align}
            h_0 &= \sigma(W_{d,0} d),\\
            h_i &= \sigma(W_{d,i} d + W_{h,i} h_{i-1}),\\
            f(d) &= W_{h,D} h_D,
        \end{align}

    where each :math:`W_{h,i} \geq 0` and :math:`\sigma` is a convex and
    monotonic :py:class:`~torch.nn.LeakyReLU`\ .
    """
    activation: Module
    r"""Activation module (:py:class:`~torch.nn.LeakyReLU`\ )."""
    hidden_weights: ParameterList
    r"""Scale of hidden weight matrices :math:`W_{h,i} \geq 0`\ ."""
    input_weights: ParameterList
    r"""List of input-injection weight matrices :math:`W_{d,i}`\ ."""
    output_weight: Parameter
    r"""Output weight vector :math:`W_{h,D} \geq 0`\ ."""

    def __init__(self,
                 depth: int,
                 width: int,
                 negative_slope: float = 0.5,
                 scale=1.0) -> None:
        r"""
        Args:
            depth: Network depth :math:`D`\ .
            width: Network width.
            negative_slope: Negative slope of LeakyReLU activation.
            scale: Length scale of object in meters.
        """
        assert 0.0 <= negative_slope < 1.0
        super().__init__()

        hidden_weights = []
        scale_hidden = 2 * (2.0 / (1 + negative_slope**2))**0.5 / width
        for _ in range(depth - 1):
            hidden_weight = 2 * (torch.rand((width, width)) - 0.5)
            hidden_weight *= scale_hidden
            hidden_weights.append(Parameter(hidden_weight, requires_grad=True))

        input_weights = []
        for layer in range(depth):
            input_weight = torch.empty((3, width))
            torch.nn.init.kaiming_uniform(input_weight)
            if layer > 0:
                input_weight *= 2**(-0.5)
            input_weights.append(Parameter(input_weight, requires_grad=True))

        scale_out = scale * 2 * (2.0 / (width * (1 + negative_slope**2)))**0.5
        output_weight = 2 * (torch.rand(width) - 0.5) * scale_out

        self.hidden_weights = ParameterList(hidden_weights)
        self.input_weights = ParameterList(input_weights)
        self.output_weight = Parameter(output_weight, requires_grad=True)
        self.activation = torch.nn.LeakyReLU(negative_slope=negative_slope)

    def abs_weights(self) -> Tuple[List[Tensor], Tensor]:
        r"""Returns non-negative version of hidden weight matrices by taking
        absolute value of :py:attr:`hidden_weights` and
        :py:attr:`output_weight`\ ."""
        abs_hidden_wts = [torch.abs(weight) for weight in self.hidden_weights]
        return abs_hidden_wts, torch.abs(self.output_weight)

    def activation_jacobian(self, activations: Tensor) -> Tensor:
        """Returns flattened diagonal Jacobian of LeakyReLU activation.

        The jacobian is simply ``1`` at indices where the activations are
        positive and :py:attr:`self.activation.negative_slope` otherwise.

        Args:
            activations: `(*, width)` output of activation function for some
              layer.

        Returns:
            `(*, width)` activation jacobian.
        """
        jacobian = torch.ones_like(activations)
        jacobian[activations <= 0] *= self.activation.negative_slope
        return jacobian

    def network_activations(self,
                            directions: Tensor) -> Tuple[List[Tensor], Tensor]:
        """Forward evaluation of the network activations

        Args:
            directions: ``(*, 3)`` network inputs.

        Returns:
            List of ``(*, width)`` hidden layer activations.
            ``(*,)`` network output
        """
        hiddens = []
        hidden_wts, output_wt = self.abs_weights()
        input_wts = self.input_weights
        # (*, 3) x (*, 3, W)
        hiddens.append(self.activation(pbmm(directions, input_wts[0])))
        # print(hiddens[-1].norm(dim=-1).mean(dim=0))
        for hidden_wt, input_wt in zip(hidden_wts, input_wts[1:]):
            linear_hidden = pbmm(hiddens[-1], hidden_wt)
            linear_input = pbmm(directions, input_wt)
            linear_output = linear_hidden + linear_input
            hiddens.append(self.activation(linear_output))
        output = pbmm(hiddens[-1], output_wt)
        return hiddens, output.squeeze(-1)

    def forward(self, directions: Tensor) -> Tensor:
        """Evaluates support function Jacobian at provided inputs.

        Args:
            directions: ``(*, 3)`` network inputs.

        Returns:
            ``(*, 3)`` network input Jacobian.
        """
        hidden_wts, output_wt = self.abs_weights()
        hiddens, _ = self.network_activations(directions)
        input_wts = self.input_weights

        hidden_jacobian = (output_wt.expand(hiddens[-1].shape) *
                           self.activation_jacobian(hiddens[-1])).unsqueeze(-1)

        jacobian = torch.zeros_like(directions)
        layer_bundle = zip(reversed(hiddens[:-1]), reversed(hidden_wts),
                           reversed(list(input_wts[1:])))

        for hidden, hidden_wt, input_wt in layer_bundle:
            jacobian += pbmm(input_wt, hidden_jacobian).squeeze(-1)

            hidden_jacobian = pbmm(hidden_wt, hidden_jacobian) * \
                self.activation_jacobian(hidden).unsqueeze(-1)

        jacobian += pbmm(input_wts[0], hidden_jacobian).squeeze(-1)

        return jacobian
