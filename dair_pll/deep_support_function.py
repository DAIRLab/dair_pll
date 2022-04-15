"""Modelling and manipulation of convex support functions."""
from typing import Callable, Tuple, List

import torch.nn
from scipy.spatial import ConvexHull  # type: ignore
from torch import Tensor
from torch.nn import Parameter, ParameterList, Module

from dair_pll.system import MeshSummary
from dair_pll.tensor_utils import pbmm

# pylint: disable=E1103
_LINEAR_SPACE = torch.linspace(-1, 1, steps=8)
_GRID = torch.cartesian_prod(_LINEAR_SPACE, _LINEAR_SPACE, _LINEAR_SPACE)
_SURFACE = _GRID[_GRID.abs().max(dim=-1).values >= 1.0]
_SURFACE = _SURFACE / _SURFACE.norm(dim=-1, keepdim=True)


def extract_mesh(support_function: Callable[[Tensor], Tensor]) -> MeshSummary:
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

    # the inward normal and a vector from the face to the centroid should
    # have positive dot product. If they do not, then the face indices are in
    # reversed order, and must be flipped.
    centroid = vertices.mean(dim=0, keepdim=True)
    v_x = vertices[faces[:, 0], :]
    v_y = vertices[faces[:, 1], :]
    v_z = vertices[faces[:, 2], :]
    inward_normals = torch.cross(v_x - v_y, v_z - v_y)
    backwards = (inward_normals * (centroid - v_x)).sum(dim=-1) < 0.

    # reorder indices of backwards faces
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
            ``(*, 1)`` network output
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
        return hiddens, output

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
        layer_bundle = zip(reversed(hiddens[:-1]),
                           reversed(hidden_wts),
                           reversed(list(input_wts[1:])))

        for hidden, hidden_wt, input_wt in layer_bundle:
            jacobian += pbmm(input_wt, hidden_jacobian).squeeze(-1)

            hidden_jacobian = pbmm(hidden_wt, hidden_jacobian) * \
                self.activation_jacobian(hidden).unsqueeze(-1)

        jacobian += pbmm(input_wts[0], hidden_jacobian).squeeze(-1)

        return jacobian
