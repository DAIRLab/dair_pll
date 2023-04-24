r"""Quaternion-based :math:`SO(3)` operations."""
from typing import TypeVar, cast

import numpy as np
import torch
from torch import Tensor
from typing_extensions import Protocol

#:
DataType = TypeVar('DataType', Tensor, np.ndarray)
r"""Static type for both supported types of quaternion/vector 
representations: :class:`~torch.Tensor` and :class:`~numpy.ndarray`\ ."""


class TensorCallable(Protocol):
    r"""Static type for callable mapping from list of :class:`~torch.Tensor`\ s
    to :class:`~torch.Tensor`\ ."""

    # pylint: disable=too-few-public-methods
    def __call__(self, *args: Tensor) -> Tensor:
        ...


class NdarrayCallable(Protocol):
    r"""Static type for callable mapping from list of :class:`~numpy.ndarray`\ s
        to :class:`~numpy.ndarray`\ ."""

    # pylint: disable=too-few-public-methods
    def __call__(self, *args: np.ndarray) -> np.ndarray:
        ...


def operation_selector(tensor_operation: TensorCallable,
                       ndarray_operation: NdarrayCallable,
                       *args: DataType) -> DataType:
    r"""Helper function which selects between Pytorch and Numpy
    implementations of a quaternion operation.

    Args:
        tensor_operation: :class:`~torch.Tensor`\ -backed implementation.
        ndarray_operation: :class:`~numpy.ndarray`\ -backed implementation.
        *args: Arguments to pass to implementation.

    Returns:
        Operation's return value, same type as arguments.
    """
    assert len(args) > 0
    if isinstance(args[0], Tensor):
        return tensor_operation(*args)

    assert isinstance(args[0], np.ndarray)
    return ndarray_operation(*args)


def inverse_torch(q: Tensor) -> Tensor:
    r""":class:`~torch.Tensor` implementation of :func:`inverse`\ ."""
    assert q.shape[-1] == 4

    q_inv = q.clone()
    q_inv[..., 1:] *= -1
    return q_inv


def inverse_np(q: np.ndarray) -> np.ndarray:
    r""":class:`~numpy.ndarray` implementation of :func:`inverse`\ ."""
    assert q.shape[-1] == 4

    q_inv = np.copy(q)
    q_inv[..., 1:] *= -1
    return q_inv


def inverse(q: DataType) -> DataType:
    r"""Quaternion inverse function.

    For input quaternion :math:`q = [q_w, q_{xyz}]`\ , returns the inverse
    quaternion :math:`q^{-1} = [-q_w, q_{xyz}]`\ .

    Args:
        q: ``(*, 4)`` quaternion batch to invert.

    Returns:
        ``(*, 4)`` inverse of ``q``.
    """
    return operation_selector(cast(TensorCallable, inverse_torch),
                              cast(NdarrayCallable, inverse_np), q)


def multiply_torch(q: Tensor, r: Tensor) -> Tensor:
    r""":class:`~torch.Tensor` implementation of :func:`multiply`\ ."""
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    q_w = q[..., :1]
    q_xyz = q[..., 1:]

    r_w = r[..., :1]
    r_xyz = r[..., 1:]

    # pylint: disable=E1103
    qr_w = q_w * r_w - torch.sum(q_xyz * r_xyz, dim=-1, keepdim=True)
    qr_xyz = q_w * r_xyz + r_w * q_xyz + torch.cross(q_xyz, r_xyz, dim=-1)

    return torch.cat((qr_w, qr_xyz), dim=-1)


def multiply_np(q, r):
    r""":class:`~numpy.ndarray` implementation of :func:`multiply`\ ."""
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    q_w = q[..., :1]
    q_xyz = q[..., 1:]

    r_w = r[..., :1]
    r_xyz = r[..., 1:]

    qr_w = q_w * r_w - np.sum(q_xyz * r_xyz, axis=-1, keepdims=True)
    qr_xyz = q_w * r_xyz + r_w * q_xyz + np.cross(q_xyz, r_xyz)

    return np.concatenate([qr_w, qr_xyz], axis=-1)


def multiply(q: DataType, r: DataType) -> DataType:
    r"""Quaternion multiplication.

    Given 2 quaternions :math:`q = [q_w, q_{xyz}]` and :math:`r = [r_w,
    r_{xyz}]`\ , performs the quaternion multiplication via the formula

    .. math::

        q \times r = \begin{bmatrix} q_w r_w - q_{xyz} \cdot r_{xyz} \\
            q_w r_{xyz} + r_w q_{xyz} + q_{xyz} \times r_{xyz}
            \end{bmatrix}

    This formula was taken from the following address:
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix

    Args:
        q: ``(*, 4)`` left quaternion factor .
        r: ``(*, 4)`` right quaternion factor.

    Returns:
        ``(*, 4)`` Product quaternion ``q * r``.
    """
    return operation_selector(cast(TensorCallable, multiply_torch),
                              cast(NdarrayCallable, multiply_np), q, r)


def rotate_torch(q: Tensor, p: Tensor) -> Tensor:
    r""":class:`~torch.Tensor` implementation of :func:`rotate`\ ."""
    assert q.shape[-1] == 4
    assert p.shape[-1] == 3

    q_w = q[..., :1]
    q_xyz = q[..., 1:]

    # pylint: disable=E1103
    q_xyz_cross_p = torch.cross(q_xyz, p, dim=-1)
    q_xyz_cross_q_xyz_cross_p = torch.cross(q_xyz, q_xyz_cross_p, dim=-1)
    q_xyz_dot_p = torch.sum(q_xyz * p, dim=-1, keepdim=True)

    return q_xyz * (q_xyz_dot_p) + q_w * (2 * q_xyz_cross_p + q_w * p) + \
           q_xyz_cross_q_xyz_cross_p


def rotate_np(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    r""":class:`~numpy.ndarray` implementation of :func:`rotate`\ ."""
    assert q.shape[-1] == 4
    assert p.shape[-1] == 3

    q_w = q[..., :1]
    q_xyz = q[..., 1:]

    q_xyz_cross_p = np.cross(q_xyz, p)
    q_xyz_cross_q_xyz_cross_p = np.cross(q_xyz, q_xyz_cross_p)
    q_xyz_dot_p = np.sum(q_xyz * p, axis=-1, keepdims=True)

    return q_xyz * (q_xyz_dot_p) + q_w * (2 * q_xyz_cross_p + q_w * p) + \
           q_xyz_cross_q_xyz_cross_p


def rotate(q: DataType, p: DataType) -> DataType:
    r"""Quaternion rotation.

       Given a quaternion :math:`q = [q_w, q_{xyz}]` and vector :math:`p`\ ,
       produces the :math:`q`\ -rotated vector :math:`p'` via the formula

       .. math::

           p' = (q_{xyz} \cdot p) q_{xyz} + 2 q_w (q_{xyz} \times p) +
           q_w^2 p + q_{xyz} \times (q_{xyz} \times p)

       This formula was taken from the following address:
       https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix

       Args:
           q: ``(*, 4)`` quaternion batch.
           p: ``(*, 3)`` vector batch.

       Returns:
           ``(*, 3)`` rotated vector batch.
       """
    return operation_selector(cast(TensorCallable, rotate_torch),
                              cast(NdarrayCallable, rotate_np), q, p)


def sinc(x: Tensor) -> Tensor:
    r"""Elementwise :math:`\mathrm{sinc}` function.

    Given a tensor :math:`x`, applies the elementwise-mapping

    .. math::

        x \to \begin{cases}1 & x =0 \\ \frac{\sin(x)}{x} & x \neq 0\end{cases}.

    Args:
        x: ``(*,)`` :math:`\mathrm{sinc}` input values.

    Returns:
        ``(*,)`` :math:`\mathrm{sinc}` function evaluated at ``x``.
    """
    # pylint: disable=E1103
    notnull = torch.abs(x) > 0
    null = torch.logical_not(notnull)
    sinc_x = torch.zeros_like(x)
    sinc_x[null] += 1.
    sinc_x[notnull] += torch.sin(x[notnull]) / (x[notnull])
    return sinc_x


def log(q: Tensor) -> Tensor:
    r"""Transforms quaternion into logarithmic coordinates.

    Given a quaternion

    .. math::

        q = [\cos(\theta/2), \hat u \sin(\theta/2)] = [q_w, q_{xyz}],

    returns the corresponding logarithmic coordinates (rotation vector)
    :math:`r = \theta\hat u`\ .

    This computation is evaluated via the pertations

    .. math::
        \begin{align}
        \theta(q) &= 2\mathrm{atan2}(||q_{xyz}||_2, q_w), \\
        r &= \begin{cases} \frac{\theta(q)}{\sin(\theta(q)/2)} q_{xyz} & \sin(
            \theta(q)/2) \neq 0,\\
            0 & \sin(\theta(q)/2) = 0.\end{cases}
        \end{align}

    This function inverts :func:`exp`.

    Args:
        q: ``(*, 4)`` quaternion batch.

    Returns:
        ``(*, 3)`` rotation vector batch :math:`r`\ .
    """
    assert q.shape[-1] == 4
    cos_half_theta = q[..., 0:1]
    q_xyz = q[..., 1:]
    sin_half_theta = torch.norm(q_xyz, dim=-1, keepdim=True)

    # pylint: disable=E1103
    theta = torch.atan2(sin_half_theta, cos_half_theta) * 2
    mul = torch.zeros_like(sin_half_theta)
    not_null = torch.abs(sin_half_theta) > 0
    mul[not_null] = theta[not_null] / sin_half_theta[not_null]

    return q_xyz * mul


def exp(r: Tensor) -> Tensor:
    r"""Transforms logarithmic coordinates into quaternion.

        Given logarithmic coordinates representation (rotation vector)
        :math:`r = \theta\hat u`\, returns the corresponding quaternion

        .. math::

            q = [\cos(\theta/2), \hat u \sin(\theta/2)] = [q_w, q_{xyz}].


        This computation is evaluated via the operations

        .. math::
            \begin{align}
            \theta(r) &= ||r||_2, \\
            q &= \begin{bmatrix}\cos(\theta(r)/2) \\
                \frac{1}{2} r \mathrm{sinc}(\theta(r)/2)
                \end{bmatrix}.
            \end{align}

        This function inverts :func:`log`.

        Args:
            r: ``(*, 3)`` rotation vector batch.

        Returns:
            ``(*, 4)`` quaternion batch :math:`q`\ .
        """
    assert r.shape[-1] == 3

    # pylint: disable=E1103
    angle = torch.norm(r, dim=-1, keepdim=True)
    return torch.cat((torch.cos(angle / 2), r * sinc(angle / 2) / 2), dim=-1)



def quaternion_to_rotmat_vec(q: Tensor) -> Tensor:
    """
    Converts batched quaternions of shape (batch, 4)
    to vectorized rotation matrices of shape (batch, 9)
    """
    qr = q[:, 0:1]
    qi = q[:, 1:2]
    qj = q[:, 2:3]
    qk = q[:, 3:4]
    r1 = torch.cat((1. - 2*(qj ** 2 + qk ** 2),
                   2*(qi*qj - qk*qr),
                   2*(qi*qk + qj*qr)), dim=1)
    r2 = torch.cat((2*(qi*qj + qk*qr),
                   1. - 2*(qi ** 2 + qk ** 2),
                   2*(qj*qk - qi*qr)), dim=1)
    r3 = torch.cat((2*(qi*qk - qj*qr),
                   2*(qj*qk + qi*qr),
                   1. - 2*(qi ** 2 + qj ** 2)), dim=1)

    return torch.cat((r1, r2, r3), dim=1)