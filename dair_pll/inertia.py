r"""Utilities for transforming representations of rigid body inertia.

The inertial parameterization of a body :math:`B` is given by 10 degrees of
freedom:

    * :math:`m` (1 DoF), the mass.
    * :math:`^{Bo}p^{Bcm}` (3 DoF), the position of the center of mass
      :math:`Bcm` relative to the body's origin::

          p_BoBcm_B == [p_x, p_y, p_z]

    * :math:`I^{B/Bcm}` (6 DoF), the symmetric 3x3 inertia dyadic about
      the body's center of mass :math:`Bcm`::

          I_BBcm_B == [[I_xx, I_xy, I_xz],[I_xy, I_yy, I_yz],[I_xz, I_yz, I_zz]]

Here we list and several formats that can be converted between each other
freely, under the assumption that their values are non-degenerate and valid (
i.e. :math:`m > 0`).

    * ``pi_cm`` is an intuitive formatting of these 10 Dof as a standard vector in
      :math:`\mathbb{R}^{10}` as::

        [m, m * p_x, m * p_y, m * p_z, I_xx, I_yy, I_zz, I_xy, I_xz, I_yz]

    * ``drake_spatial_inertia_vector`` is a scaling that can be used to
      construct a new Drake :py:attr:`~pydrake.multibody.tree.SpatialInertia
      `, where the inertial tensor is normalized by mass (see
      :py:attr:`~pydrake.multibody.tree.UnitInertia`)::

        [m, p_x, p_y, p_z, ...
            I_xx , I_yy , I_zz , I_xy, I_xz, I_yz]

    * ``drake`` is a packaging of ``drake_inertia_vector`` into
      a Drake :py:attr:`~pydrake.multibody.tree.SpatialInertia` object,
      with member callbacks to access the terms::

        SpatialInertia.get_mass() == m
        SpatialInertia.get_com() == p_BoBcm_B
        SpatialInertia.CalcRotationalInertia() == I_BBo_B
        SpatialInertia.Shift(p_BoBcm_B).CalcRotationalInertia() == I_BBcm_B

    * ``theta`` is a format designed for underlying smooth, unconstrained,
      and non-degenerate parameterization of rigid body inertia. For a body,
      any value in :math:`\mathbb{R}^{10}` for ``theta`` can be mapped to a
      valid and non-degenerate set of inertial terms as follows::

        theta == [log_m, h_x, h_y, h_z, d_1, d_2, d_3, s_12, s_13, s_23]
        m == exp(log_m)
        p_BoBcm_B == [h_x, h_y, h_z] / m
        I_BBcm_B = trace(Sigma(theta)) * I_3 - Sigma(theta)

      where ``Sigma`` :math:`(\Sigma)` is constructed via log-Cholesky
      parameterization, similar to the one in  Rucker and Wensing [1]_ :

      .. math::
        \begin{align}
        \Sigma &= L L^T, \\
        L &= \begin{bmatrix} \exp(d_1) & 0 & 0 \\
                          s_{12} & \exp(d_2) & 0 \\
                          s_{13} & s_{23} & \exp(d_3)
                          \end{bmatrix}.
        \end{align}


      While this parameterization is distinct, it retains the diffeomorphism
      between :math:`\theta \in \mathbb{R}^{10}` and valid rigid body
      inertia. Note that we use the Drake ordering of the inertial
      off-diagonal terms as ``[Ixy Ixz Iyz]``, whereas Rucker and Wensing
      [1]_ uses ``[Ixy Iyz Ixz]``.

    * ``urdf`` is the string format in which inertial parameters are stored,
      represented as the tuple::

        "m", "p_x p_y p_z", ["I_xx", "I_yy", "I_zz", "I_xy", "I_xz", "I_yz"]

    * ``scalars`` is the string dictionary format for printing on tensorboard::

        {"m": m, "p_x": p_x, ... "I_yz": I_yz}

Various transforms between these types are implemented in this module through
the :py:class:`InertialParameterConverter` class.

.. [1] C. Rucker and P. M. Wensing, "Smooth Parameterization of Rigid-Body
    Inertia", IEEE RA-L 2020, https://doi.org/10.1109/LRA.2022.3144517
"""
from typing import Any, Tuple, List, Dict

import torch
from torch import Tensor

from dair_pll.drake_utils import DrakeSpatialInertia
from dair_pll.tensor_utils import skew_symmetric, symmetric_offdiagonal, \
    pbmm, trace_identity

torch.set_default_dtype(torch.float64)  # pylint: disable=no-member

INERTIA_INDICES = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
INERTIA_SCALARS = ["I_xx", "I_yy", "I_zz", "I_xy", "I_xz", "I_yz"]
AXES = ["x", "y", "z"]


def number_to_float(number: Any) -> float:
    """Converts a number to float via intermediate string representation."""
    return float(str(number))

# pylint: disable=invalid-name
def parallel_axis_theorem(I_BBa_B: Tensor,
                          m_B: Tensor,
                          p_BaBb_B: Tensor,
                          Ba_is_Bcm: bool = True) -> Tensor:
    """Converts an inertia matrix represented from one reference point to that
    represented from another reference point.  One of these reference points
    must be the center of mass.

    The parallel axis theorem states [2]:

    .. math::

        I_R = I_C - m_{tot} [d]^2


    ...for :math:`I_C` as the inertia matrix about the center of mass,
    :math:`I_R` as the moment of inertia about a point :math:`R` defined as
    :math:`R = C + d`, and :math:`m_{tot}` as the total mass of the body.  The
    brackets in :math:`[d]` indicate the skew-symmetric matrix formed from the
    vector :math:`d`.

    [2] https://en.wikipedia.org/wiki/Moment_of_inertia#Parallel_axis_theorem

    Args:
        I_BBa_B: ``(*, 3, 3)`` inertia matrices.
        m_B: ``(*)`` masses.
        p_BaBb_B: ``(*, 3)`` displacement from current frame to new frame.
        Ba_is_Bcm: ``True`` if the provided I_BBa_B is from the perspective of
          the CoM, ``False`` if from the perspective of the origin.

    Returns:
        ``(*, 3, 3)`` inertia matrices with changed reference point.
    """
    d_squared = skew_symmetric(p_BaBb_B) @ skew_symmetric(p_BaBb_B)
    term = d_squared * m_B.view((-1, 1, 1))

    if Ba_is_Bcm:
        return I_BBa_B - term

    return I_BBa_B + term
# pylint: enable=invalid-name

def inertia_matrix_from_vector(I_BBa_B_vec: Tensor) -> Tensor:
    r"""Converts vectorized inertia vector of the following order into an
    inertia matrix:

    .. math::

        [I_{xx}, I_{yy}, I_{zz}, I_{xy}, I_{xz}, I_{yz}] \Rightarrow
        \begin{bmatrix} I_{xx} & I_{xy} & I_{xz} \\
        I_{xy} & I_{yy} & I_{yz} \\
        I_{xz} & I_{yz} & I_{zz} \end{bmatrix}

    Args:
        I_BBa_B_vec: ``(*, 6)`` vectorized inertia parameters.

    Returns:
        ``(*, 3, 3)`` inertia matrix.
    """
    # Put Ixx, Iyy, Izz on the diagonals.
    diags = torch.diag_embed(I_BBa_B_vec[..., :3])

    # Put Ixy, Ixz, Iyz on the off-diagonals.
    off_diags = symmetric_offdiagonal(I_BBa_B_vec[..., 3:].flip(-1))

    return diags + off_diags


def inertia_vector_from_matrix(I_BBa_B_mat: Tensor) -> Tensor:
    r"""Converts inertia matrix into vectorized inertia vector of the following
    order:

    .. math::

        \begin{bmatrix} I_{xx} & I_{xy} & I_{xz} \\
        I_{xy} & I_{yy} & I_{yz} \\
        I_{xz} & I_{yz} & I_{zz} \end{bmatrix} \Rightarrow
        [I_{xx}, I_{yy}, I_{zz}, I_{xy}, I_{xz}, I_{yz}]
    Args:
        I_BBa_B_mat: ``(*, 3, 3)`` inertia matrix.

    Returns:
        ``(*, 6)`` vectorized inertia parameters.
    """
    # Grab Ixx, Iyy, Izz on the diagonals.
    diagonals = I_BBa_B_mat.diagonal(dim1=-2, dim2=-1)

    # Grab Ixy, Ixz, Iyz on the off-diagonals individually.
    I_xy = I_BBa_B_mat[..., 0, 1]
    I_xz = I_BBa_B_mat[..., 0, 2]
    I_yz = I_BBa_B_mat[..., 1, 2]

    offdiagonals = torch.stack((I_xy, I_xz, I_yz), dim=-1)

    return torch.cat((diagonals, offdiagonals), dim=1)


class InertialParameterConverter:
    """Utility class for transforming between inertial parameterizations."""

    @staticmethod
    def theta_to_pi_cm(theta: Tensor) -> Tensor:
        """Converts batch of ``theta`` parameters to ``pi_cm`` parameters.

        Args:
            theta: ``(*, 10)`` ``theta``-type parameterization.

        Returns:
            ``(*, 10)`` ``pi_cm``-type parameterization.
        """
        mass = torch.exp(theta[..., :1])
        h_vector = theta[..., 1:4]
        d_vector = theta[..., 4:7]
        s_vector = theta[..., 7:]

        diagonal_exp_d = torch.diag_embed(torch.exp(d_vector))

        # lower-triangular component of symmetrized
        lower_triangular_s = symmetric_offdiagonal(s_vector.flip(1)).tril()

        cholesky_sigma = diagonal_exp_d + lower_triangular_s

        sigma = pbmm(cholesky_sigma, cholesky_sigma.mT)

        I_BBcm_B = trace_identity(sigma) - sigma

        I_BBcm_B_vec = inertia_vector_from_matrix(I_BBcm_B)

        return torch.cat((mass, h_vector, I_BBcm_B_vec), dim=-1)

    @staticmethod
    def pi_cm_to_theta(pi_cm: Tensor) -> Tensor:
        """Converts batch of ``pi_cm`` parameters to ``theta`` parameters.

        Implements local inverse :py:meth:`theta_to_pi_cm` for valid ``pi_cm``.

        Args:
            pi_cm: ``(*, 10)`` ``pi_cm``-type parameterization.

        Returns:
            ``(*, 10)`` ``theta``-type parameterization.
        """

        log_m = torch.log(pi_cm[..., :1])

        h_vector = pi_cm[..., 1:4]

        I_BBcm_B = inertia_matrix_from_vector(pi_cm[..., 4:])

        sigma = 0.5 * trace_identity(I_BBcm_B) - I_BBcm_B

        cholesky_sigma = torch.linalg.cholesky(sigma)

        d_vector = torch.log(torch.diagonal(cholesky_sigma, dim1=-2, dim2=-1))

        s_vector = torch.stack(
            (cholesky_sigma[..., 1, 0], cholesky_sigma[..., 2, 0],
             cholesky_sigma[..., 2, 1]),
            dim=-1)

        return torch.cat((log_m, h_vector, d_vector, s_vector), -1)

    @staticmethod
    def pi_cm_to_drake_spatial_inertia_vector(pi_cm: Tensor) -> Tensor:
        """Converts batch of ``pi-cm`` parameters to ``drake_inertia_vector``
        parameters."""
        return torch.cat((pi_cm[..., 0:1],
                          pi_cm[..., 1:4] / pi_cm[..., 0:1],
                          pi_cm[..., 4:]),
                         dim=-1)

    @staticmethod
    def pi_cm_to_urdf(pi_cm: Tensor) -> Tuple[str, str, List[str]]:
        """Converts a single ``(10,)`` ``pi_cm`` vector into the ``urdf`` string
        format."""
        assert len(pi_cm.shape) == 1
        mass = str(pi_cm[0].item())
        p_BoBcm_B = ' '.join(
            [str((coordinate / pi_cm[0]).item()) for coordinate in pi_cm[1:4]])
        I_BBcm_B = [
            str(inertia_element.item()) for inertia_element in pi_cm[4:]
        ]

        return mass, p_BoBcm_B, I_BBcm_B

    @staticmethod
    def drake_to_pi_cm(M_BBo_B: DrakeSpatialInertia) -> Tensor:
        """Extracts a ``pi_cm`` parameterization from a Drake
        :py:attr:`~pydrake.multibody.tree.SpatialInertia` object.

        Args:
            M_BBo_B: Drake spatial inertia of body, about body origin, in body
              coordinates.
        """
        mass = number_to_float(M_BBo_B.get_mass())
        p_BoBcm_B = M_BBo_B.get_com()
        M_BBcm_B = M_BBo_B.Shift(p_BoBcm_B)
        I_BBcm_B = M_BBcm_B.CalcRotationalInertia()

        mass_list = [
            mass * number_to_float(coordinate) for coordinate in p_BoBcm_B
        ]

        inertia_list = [
            number_to_float(I_BBcm_B[index[0], index[1]])
            for index in INERTIA_INDICES
        ]
        pi = Tensor([mass] + mass_list + inertia_list)
        return pi

    @staticmethod
    def drake_to_theta(M_BBo_B: DrakeSpatialInertia) -> Tensor:
        """Passthrough chain of :py:meth:`drake_to_pi_cm` and
        :py:meth:`pi_cm_to_theta`."""
        pi_cm = InertialParameterConverter.drake_to_pi_cm(M_BBo_B)
        return InertialParameterConverter.pi_cm_to_theta(pi_cm)

    @staticmethod
    def pi_cm_to_scalars(pi_cm: Tensor) -> Dict[str, float]:
        """Converts ``pi_cm`` parameterization to ``scalars`` dictionary."""
        mass = pi_cm[0]
        p_BoBcm_B = pi_cm[1:4] / mass
        I_BBcm_B = pi_cm[4:]
        scalars = {"m": mass.item()}
        scalars.update({
            f'com_{axis}': p_axis.item()
            for axis, p_axis in zip(AXES, p_BoBcm_B)
        })
        scalars.update({
            inertia_scalar: inertial_value.item()
            for inertia_scalar, inertial_value in zip(INERTIA_SCALARS, I_BBcm_B)
        })
        return scalars