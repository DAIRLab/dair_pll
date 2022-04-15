"""Tensor utility functions.

Contains various utility functions for common tensor operations required
throughout the package. All such future functions should be placed here,
with the following exceptions:

    * Utilities for operating directly on :math:`SO(3)` should be placed in
      :py:mod:`dair_pll.quaternion`
"""
from typing import List

import torch
from torch import Tensor


def tile_dim(tiling_tensor: Tensor, copies: int, dim: int = 0) -> Tensor:
    """Tiles tensor along specified dimension.

    Args:
        tiling_tensor: ``(n_0, ..., n_{k-1})`` tensor.
        copies: number of copies, ``copies >= 1``.
        dim: dimension to be tiled, ``-k <= dim <= k - 1``.

    Returns:
        ``(n_0, ..., n * n_dim, ... n_{k-1})`` tiled tensor.

    Raises:
        ValueError: when ``copies`` is not a strictly-positive integer

    """
    if not copies >= 1:
        raise ValueError(
            f'Tiling count should be positive int, got {copies} instead.')

    # pylint: disable=E1103
    return torch.cat([tiling_tensor] * copies, dim=dim)


def tile_last_dim(tiling_tensor: Tensor, copies: int) -> Tensor:
    """Tile right dimension (``-1``) via :py:func:`tile_dim`"""
    return tile_dim(tiling_tensor, copies, -1)


def tile_penultimate_dim(tiling_tensor: Tensor, copies: int) -> Tensor:
    """Tile second-to-last dimension (``-2``) :py:func:`tile_dim`"""
    return tile_dim(tiling_tensor, copies, -2)


def pbmm(t_1: Tensor, t_2: Tensor) -> Tensor:
    """Multiplies matrices with optional batching.

    Wrapper function that performs a final-axes (``-2,-1``) matrix-matrix,
    vector-matrix, matrix-vector, or vector-vector product depending on the
    shape of ``t_1`` and ``t_2``. The following logic is used:

        * do a matrix-matrix multiplication if both factors have dimension at
          least two, and broadcast to the larger (inferred) batch.
        * do a vector-matrix / matrix-vector multiplication if one factor is
          a vector and the other has dimension ``>= 2``
        * do a vector-vector multiplication if both factors are vectors.


    Args:
        t_1: ``(*, l, m)`` or ``(l, m)`` or ``(m,)`` left tensor factor.
        t_2: ``(*, m, n)`` or ``(m, n)`` or ``(m,)`` right tensor factor.

    Returns:
        ``(*, l, n) if l, n > 1 or (*, l) if l > 1, n = 1 or (*, n)
        if l = 1, or n > 1`` or scalar ``if dim(t_1) == dim(t_2) == 1`` product
        tensor.
    """
    t_1_dim = t_1.dim()
    t_2_dim = t_2.dim()
    needs_squeeze = None
    # case 1: single dot product
    if max(t_1_dim, t_2_dim) == 1:
        # dot product
        return (t_1 * t_2).sum()

    # temporarily expand dimension for vector-matrix product
    if t_1_dim == 1:
        t_1 = t_1.unsqueeze(0)
        t_1_dim = 2
        needs_squeeze = -2
    elif t_2_dim == 1:
        t_2 = t_2.unsqueeze(1)
        t_2_dim = 2
        needs_squeeze = -1

    # cases 2 and 3: matrix product
    if max(t_1_dim, t_2_dim) > 2:
        # match batching
        if t_1_dim < t_2_dim:
            t_1 = t_1.expand(t_2.shape[:-t_1_dim] + t_1.shape)
        elif t_1_dim > t_2_dim:
            t_2 = t_2.expand(t_1.shape[:-t_2_dim] + t_2.shape)

        # pylint: disable=E1103
        product = torch.matmul(t_1, t_2)
    else:
        product = t_1.mm(t_2)

    if needs_squeeze:
        product = product.squeeze(needs_squeeze)
    return product


def deal(dealing_tensor: Tensor,
         dim: int = 0,
         keep_dim: bool = False) -> List[Tensor]:
    """Converts dim of tensor to list.

    Example:
        Let ``t`` be a 3-dimensional tensor of shape ``(3,5,3)`` such that::

            t[:, i, :] == torch.eye(3).

        Then ``deal(t, dim=1)`` returns a list of 5 ``(3,3)`` identity tensors,
        and ``deal(t, dim=1, keep_dim=True)`` returns a list of ``(3,1,3)``
        tensors.

    Args:
        dealing_tensor: ``(n_0, ..., n_dim, ..., n_{k-1})`` shaped tensor.
        dim: tensor dimension to deal, ``-k <= dim <= k-1``.
        keep_dim: whether to squeeze list items along ``dim``.

    Returns:
        List of dealt sub-tensors of shape ``(..., n_{dim-1}, {n_dim+1}, ...)``
        or ``(..., n_{dim-1}, 1, {n_dim+1}, ...)``.
    """
    tensor_list = torch.split(dealing_tensor, 1, dim=dim)
    if keep_dim:
        return tensor_list
    return [tensor_i.squeeze(dim) for tensor_i in tensor_list]


def skew_symmetric(vectors: Tensor) -> Tensor:
    r"""Converts vectors in :math:`\mathbb{R}^3` into skew-symmetric form.

    Converts vector(s) :math:`v` in ``vectors`` into skew-symmetric matrix:

    .. math::

        S(v) = -S(v)^T = \begin{bmatrix} 0 & -v_3 & v_2 \\
        v_3 & 0 & -v_1 \\
        -v_2 & v_1 & 0 \end{bmatrix}

    Args:
        vectors: ``(*, 3)`` vector(s) to convert to matrices

    Returns:
        ``(*, 3, 3)`` skew-symmetric matrices :math:`S(v)`
    """
    # pylint: disable=E1103
    zero = torch.zeros_like(vectors[..., 0])

    # pylint: disable=E1103
    row_1 = torch.stack((zero, -vectors[..., 2], vectors[..., 1]), -1)
    row_2 = torch.stack((vectors[..., 2], zero, -vectors[..., 0]), -1)
    row_3 = torch.stack((-vectors[..., 1], vectors[..., 0], zero), -1)

    return torch.stack((row_1, row_2, row_3), -2)


def batch_diagonal(vectors: Tensor) -> Tensor:
    """Converts vectors to diagonal matrices.

    Take in an arbitrary batch of n-vectors and returns the same sized batch
    of (n, n) matrices such that::

        output[b_1, ..., b_k, :, :] == torch.diag(vectors[b_1, ..., b_k, :]).

    Code structure comes from thw following address:

        https://discuss.pytorch.org/t/batch-of-diagonal-matrix/13560

    Args:
        vectors: ``(*, n)`` batch of

    Returns:
        ``(*, n, n)`` batch of diagonal matrices
    """
    # make a zero matrix, which duplicates the last dim of input
    dims = vectors.shape + (vectors.shape[-1],)
    # pylint: disable=E1103
    output = torch.zeros(dims)

    # stride across the first dimensions, and add one to get the diagonal of the
    # last dimension
    # pylint: disable=E1103
    strides = [output.stride(i) for i in torch.arange(vectors.dim() - 1)]
    strides.append(output.shape[-1] + 1)

    # stride and copy the input to the diagonal
    output.as_strided(vectors.size(), strides).copy_(vectors)
    return output


def one_vector_block_diagonal(num_blocks: int, vector_length: int) -> Tensor:
    """Computes a block diagonal matrix with column vectors of ones as blocks.

    Associated with the mathematical symbol :math:`E`.

    Example:
        ::

            one_vector_block_diagonal(3, 2) == tensor([
                [1., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [0., 0., 1.]]).

    Args:
        num_blocks: number of columns.
        vector_length: number of ones in each matrix diagonal block.

    Returns:
        ``(n * vector_length, n)`` 0-1 tensor.
    """
    # pylint: disable=E1103
    return torch.eye(num_blocks).repeat(1, vector_length).reshape(
        num_blocks * vector_length, num_blocks)


def spatial_to_point_jacobian(p_BoP_E: Tensor) -> Tensor:
    r"""Body-fixed translational velocity to spatial velocity Jacobian.

    Takes a batch of points :math:`[^{Bo}p^P]_E` fixed to body :math:`B` and
    expressed in some coordinates :math:`E`, and constructs the Jacobian of
    their linear velocity in some other frame :math:`A` w.r.t. the
    :math:`E`-coordinate spatial velocity of :math:`B` relative to :math:`A`.

    In detail, let the :math:`i`th element of the batch represent point ``Pi``
    as :math:`[^{Bo}p^{Pi}]_E`, and let Ao be fixed in A. The Jacobian
    calculated is

    .. math::

        J = \frac{\partial [^Av^{Pi}]_E }{\partial [^AV^B]_E}.

    We have that :math:`[^AV^B]_E = [^A\omega^B; ^{A}v^{Bo}]_E`, and from
    kinematics that

    .. math::

        ^Av^{Pi}= ^{A}v^{Bo} + ^A\omega^B \times ^{Bo}p^{Pi}.

    Thus, the Jacobian is of the form

    .. math::

       J = [-S([^{Bo}p^{Pi}]_E), I_3],

    where :math:`S` is calculated via :py:func:`skew_symmetric`.

    Args:
        p_BoP_E: ``(*, 3)``, body frame point(s) :math:`P` in coordinates
          :math:`E`

    Returns:
        ``(*, 3, 6)`` Jacobian tensor(s) :math:`J`

    """
    left = -skew_symmetric(p_BoP_E)

    # pylint: disable=E1103
    right = torch.eye(3).expand(p_BoP_E.shape[:-1] + (3, 3))

    # pylint: disable=E1103
    return torch.cat((left, right), dim=-1)


def rotation_matrix_from_one_vector(directions: Tensor, axis: int) -> Tensor:
    r"""Converts a batch of directions for specified axis, to a
    batch of rotation matrices.

    Specifically, if the ``i``\ th provided direction is ``d_i``, then the
    ``i``\ th returned rotation matrix ``R_i`` obeys::

        R_i[:, axis] == d_i.

    Reimplements the algorithm from Drake's
    :py:meth:`pydrake.math.RotationMatrix_[float].MakeFromOneVector`. For more
    details,
    see ``rotation_matrix.cc:L13`` at the following address:

        https://github.com/RobotLocomotion/drake/blob/d9c453d214ef715c89ab0e8553cae24900b7adde/math/rotation_matrix.cc#L13

    Args:
        directions: ``(*, 3)`` x/y/z directions.
        axis: ``0``, ``1``, or ``2`` depending on if ``directions`` are x, y, or
          z.
    Returns:
        ``(*, 3, 3)`` rotation matrix batch.
    """
    assert axis in [0, 1, 2]
    original_shape = directions.shape
    directions = directions.view(-1, 3)
    # pylint: disable=E1103
    batch_range = torch.arange(directions.shape[0])

    column_a = directions / directions.norm(dim=-1, keepdim=True)

    # pylint: disable=E1103
    min_a = torch.abs(directions).min(dim=-1)
    min_magnitude_a = min_a.values
    axis_i = min_a.indices
    axis_j = axis_i + 1 % 3
    axis_k = axis_j + 1 % 3

    # pylint: disable=E1103
    magnitude_a_u = torch.sqrt(1 - min_magnitude_a * min_magnitude_a)
    axis_c_correction = -min_magnitude_a / magnitude_a_u

    # pylint: disable=E1103
    column_b = torch.zeros_like(column_a)
    column_b[batch_range,
             axis_j] += -column_a[batch_range, axis_k] / magnitude_a_u
    column_b[batch_range,
             axis_j] += column_a[batch_range, axis_j] / magnitude_a_u

    column_c = torch.zeros_like(column_a)
    column_c[batch_range, axis_i] += magnitude_a_u
    column_c[batch_range,
             axis_j] += axis_c_correction * column_a[batch_range, axis_j]
    column_c[batch_range,
             axis_k] += axis_c_correction * column_a[batch_range, axis_k]

    columns = [torch.tensor(0.)] * 3
    columns[axis] = column_a
    columns[(axis + 1) % 3] = column_b
    columns[(axis + 2) % 3] = column_c

    return torch.stack(columns, dim=-1).reshape(original_shape + (3,))

def broadcast_lorentz(vectors: Tensor) -> Tensor:
    r"""Utility function that broadcasts scalars into Lorentz product cone
    format.

    This function maps a given vector :math:`v = [v_1, \dots, v_n]` in given
    batch ``vectors`` to

    .. math::

        \begin{bmatrix} v & v_1 & v_1 & \cdots & v_n & v_n \end{bmatrix}.

    Args:
        vectors: ``(*, n)`` vectors to be broadcasted.
    Returns:
        ``(*, 3 * n)`` broadcasted vectors.
    """
    n_cones = vectors.shape[-1]
    double_vectors_shape = vectors.shape[:-1] + (2 * n_cones,)
    vectors_tiled = vectors.unsqueeze(-1).repeat([1] * len(vectors.shape) +
                                             [2]).reshape(double_vectors_shape)
    # pylint: disable=E1103
    return torch.cat((vectors, vectors_tiled), dim=-1)

def project_lorentz(vectors: Tensor) -> Tensor:
    r"""Utility function that projects vectors in Lorentz cone product.

        This function takes in a batch of vectors

        .. math::

            \begin{align}
            v &= \begin{bmatrix} v_{n1} & \cdots v_{nk} & v_{t1} & \cdots v_{tk}
            \end{bmatrix},\\
            v_{ni} &\in \mathbb{R},\\
            v_{ti} &\in \mathbb{R}^2,\\
            \end{align}

        and projects each :math:`v_i = [v_{ni} v_{ti}]` into the Lorentz cone
        :math:`L = \{ v_{ni} \geq ||v_{ti}||_2\}` via the following piecewise
        formula:

            * if :math:`v_i \in L`, it remains the same.
            * if :math:`v_i \in L^{\circ} = \{-v_{ni} \geq ||v_{ti}||_2\}` (the
              polar cone), replace it with :math:`0`.
            * if :math:`v_i \not\in L \cup L^\circ`, replace it with

              .. math::

                v = \begin{bmatrix} n & \frac{n}{||v_{ti}||_2}v_{ti}
                \end{bmatrix},

              where :math:`n = \frac{1}{2}(v_{ni} + ||v_{ti}||_2)`.


        Args:
            vectors: ``(*, 3 * n)`` vectors to be projected.
        Returns:
            ``(*, 3 * n)`` broadcasted vectors.
        """
    assert vectors.shape[-1] % 3 == 0
    n_cones = vectors.shape[-1] // 3

    normals = vectors[..., :n_cones]
    tangents = vectors[..., n_cones:]
    tangent_vectors_shape = tangents.shape[:-1] + (n_cones, 2)
    tangent_norms = tangents.reshape(tangent_vectors_shape).norm(dim=-1)

    not_in_lorentz_cone = tangent_norms > normals
    in_polar_cone = tangent_norms <= -normals
    in_neither_cone = (~in_polar_cone) & not_in_lorentz_cone

    in_polar_mask = broadcast_lorentz(in_polar_cone)
    in_neither_mask = broadcast_lorentz(in_neither_cone)

    projected_vectors = vectors.clone()

    projected_vectors[in_polar_mask] *= 0.

    normals_rescaled = ((normals + tangent_norms) / 2)
    tangent_normalizer = normals_rescaled / tangent_norms
    tangent_rescaled = tangents * tangent_normalizer.unsqueeze(-1).expand(
        tangent_vectors_shape).reshape(tangents.shape)
    vectors_rescaled = torch.cat((normals_rescaled,
                                  tangent_rescaled), dim=-1)

    projected_vectors[in_neither_mask] = vectors_rescaled[in_neither_mask]

if __name__ == '__main__':
    vectors = torch.rand((100,9)) - 0.5
    project_lorentz(vectors)


