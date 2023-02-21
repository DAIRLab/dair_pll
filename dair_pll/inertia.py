"""Utilities for transforming representations of rigid body inertia.

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

    * ``pi_o`` is nearly the same as ``pi_cm`` except that the moments of inertia
      are about the body's origin :math:`Bo` instead of the center of mass.

    * ``drake_spatial_inertia`` is a scaling that can be used to construct a
      new Drake :py:attr:`~pydrake.multibody.tree.SpatialInertia`, where the
      inertial tensor is normalized by mass (see
      :py:attr:`~pydrake.multibody.tree.UnitInertia`)::

        [m, p_x, p_y, p_z, ...
            I_xx / m, I_yy / m, I_zz / m, I_xy / m, I_xz / m, I_yz / m]

    * ``drake`` is a packaging of ``drake_spatial_inertia`` into
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

        theta == [alpha, d_1, d_2, d_3, s_12, s_23, s_13, t_1, t_2, t_3]
        s == [s_12, s_23, s_13]
        t == [t_1, t_2, t_3]
        pi_o == [
            t \\cdot t + 1,
            t_1 * exp(d_1),
            t_1 * s_12 + t_2 * exp(d_2),
            t_1 * s_13 + t_2 * s_23 + t_3 * exp(d_3),
            s \\cdot s + exp(2 * d_2) + exp(2 * d_3),
            s_13 ** 2 + s_23 ** 2 + exp(2 * d_1) + exp(2 * d_3),
            s_12 ** 2 + exp(2 * d_1) + exp(2 * d_2),
            -s_12 * exp(d_1),
            -s_13 * exp(d_1),
            -s_12 * s_13 - s_23 * exp(d_2)
        ]

    An original derivation and characterization of ``theta`` can be found in
    Rucker and Wensing [1]_. Note that Drake orders the inertial off-diagonal
    terms as ``[Ixy Ixz Iyz]``, whereas the original paper [1]_ uses
    ``[Ixy Iyz Ixz]``; thus the index ordering here is slightly different.

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
import pdb

from dair_pll.drake_utils import DrakeSpatialInertia
from dair_pll.tensor_utils import deal, skew_symmetric, symmetric_offdiagonal


torch.set_default_dtype(torch.float64)

INERTIA_INDICES = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
INERTIA_SCALARS = ["I_xx", "I_yy", "I_zz", "I_xy", "I_xz", "I_yz"]
AXES = ["x", "y", "z"]


def number_to_float(number: Any) -> float:
    """Converts a number to float via intermediate string representation."""
    return float(str(number))


def parallel_axis_theorem(inertia_mat: Tensor, masses: Tensor, vec: Tensor,
                          from_com_to_other: bool = True) -> Tensor:
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
        inertia_mat: ``(*, 3, 3)`` inertia matrices.
        masses: ``(*)`` masses.
        vec: ``(*, 3)`` displacement from current frame to new frame.
        from_com_to_other: ``True`` if the provided inertia_mat is from the
          perspective of the CoM, ``False`` if from the perspective of the
          origin.

    Returns:
        ``(*, 3, 3)`` inertia matrices with changed reference point.
    """
    d_squared = skew_symmetric(vec) @ skew_symmetric(vec)
    term = d_squared * masses.view((-1, 1, 1))

    if from_com_to_other:
        return inertia_mat - term
    else:
        return inertia_mat + term


def inertia_matrix_from_vector(inertia_vec: Tensor) -> Tensor:
    r"""Converts vectorized inertia vector of the following order into an
    inertia matrix:

    .. math::

        [I_{xx}, I_{yy}, I_{zz}, I_{xy}, I_{xz}, I_{yz}] \Rightarrow
        \begin{bmatrix} I_{xx} & I_{xy} & I_{xz} \\
        I_{xy} & I_{yy} & I_{yz} \\
        I_{xz} & I_{yz} & I_{zz} \end{bmatrix}

    Args:
        inertia_vec: ``(*, 6)`` vectorized inertia parameters.

    Returns:
        ``(*, 3, 3)`` inertia matrix.
    """
    # Put Ixx, Iyy, Izz on the diagonals.
    diags = torch.diag_embed(inertia_vec[:, :3])

    # Put Ixy, Ixz, Iyz on the off-diagonals.
    off_diags = symmetric_offdiagonal(inertia_vec[:, 3:].flip(1))

    return diags + off_diags


def inertia_vector_from_matrix(inertia_mat: Tensor) -> Tensor:
    r"""Converts inertia matrix into vectorized inertia vector of the following
    order:

    .. math::

        \begin{bmatrix} I_{xx} & I_{xy} & I_{xz} \\
        I_{xy} & I_{yy} & I_{yz} \\
        I_{xz} & I_{yz} & I_{zz} \end{bmatrix} \Rightarrow
        [I_{xx}, I_{yy}, I_{zz}, I_{xy}, I_{xz}, I_{yz}]
    
    Args:
        inertia_mat: ``(*, 3, 3)`` inertia matrix.

    Returns:
        ``(*, 6)`` vectorized inertia parameters.
    """
    # Grab Ixx, Iyy, Izz on the diagonals.
    firsts = inertia_mat.diagonal(dim1=1, dim2=2)

    # Grab Ixy, Ixz, Iyz on the off-diagonals individually.
    ixys = inertia_mat[:, 0, 1].reshape(-1, 1)
    ixzs = inertia_mat[:, 0, 2].reshape(-1, 1)
    iyzs = inertia_mat[:, 1, 2].reshape(-1, 1)

    return torch.cat((firsts, ixys, ixzs, iyzs), dim=1)


class InertialParameterConverter:
    """Utility class for transforming between inertial parameterizations."""

    @staticmethod
    def theta_to_pi_o(theta: Tensor) -> Tensor:
        """Converts batch of ``theta`` parameters to ``pi_o`` parameters.

        Args:
            theta: ``(*, 10)`` ``theta``-type parameterization.

        Returns:
            ``(*, 10)`` ``pi_o``-type parameterization.
        """
        (alpha, d_1, d_2, d_3, s_12, s_23, s_13, t_1, t_2,
         t_3) = deal(theta, -1)

        s_dot_s = (theta[..., 4:7] * theta[..., 4:7]).sum(dim=-1)
        t_dot_t = (theta[..., 7:10] * theta[..., 7:10]).sum(dim=-1)

        # pylint: disable=E1103
        scaled_pi_elements = (t_dot_t + 1, t_1 * torch.exp(d_1),
                              t_1 * s_12 + t_2 * torch.exp(d_2),
                              t_1 * s_13 + t_2 * s_23 + t_3 * torch.exp(d_3),
                              s_dot_s + torch.exp(2 * d_2) + torch.exp(2 * d_3),
                              s_13 * s_13 + s_23 * s_23 + torch.exp(2 * d_1) +
                              torch.exp(2 * d_3), s_12 * s_12 +
                              torch.exp(2 * d_1) + torch.exp(2 * d_2),
                              -s_12 * torch.exp(d_1), -s_13 * torch.exp(d_1),
                              -s_12 * s_13 - s_23 * torch.exp(d_2))

        # pylint: disable=E1103
        return torch.exp(2 * alpha).unsqueeze(-1) * torch.stack(
            scaled_pi_elements, dim=-1)

    @staticmethod
    def pi_o_to_theta(pi_o: Tensor) -> Tensor:
        """Converts batch of ``pi_o`` parameters to ``theta`` parameters.

        Implements hand-derived local inverse of standard mapping from Rucker
        and Wensing.

        This function inverts :py:meth:`theta_to_pi_o` for
        valid ``pi_o``.

        Args:
            pi_o: ``(*, 10)`` ``pi_o``-type parameterization.

        Returns:
            ``(*, 10)`` ``theta``-type parameterization.
        """

        # exp(alpha)exp(d_1)
        # pylint: disable=E1103
        exp_alpha_exp_d_1 = torch.sqrt(0.5 *
                                       (pi_o[..., 5] + pi_o[..., 6] - pi_o[..., 4]))

        # exp(alpha)s_12
        exp_alpha_s_12 = -pi_o[..., 7] / exp_alpha_exp_d_1

        # exp(alpha)s_13
        exp_alpha_s_13 = -pi_o[..., 8] / exp_alpha_exp_d_1

        # exp(alpha)exp(d_2)
        # pylint: disable=E1103
        exp_alpha_exp_d_2 = torch.sqrt(pi_o[..., 6] - exp_alpha_exp_d_1 ** 2 -
                                       exp_alpha_s_12 ** 2)

        # exp(alpha)s_23
        exp_alpha_s_23 = (-pi_o[..., 9] -
                          exp_alpha_s_12 * exp_alpha_s_13) / exp_alpha_exp_d_2

        # exp(alpha)exp(d3)
        # pylint: disable=E1103
        exp_alpha_exp_d_3 = torch.sqrt(pi_o[..., 5] - exp_alpha_exp_d_1 ** 2 -
                                       exp_alpha_s_13 ** 2 - exp_alpha_s_23 ** 2)

        # exp(alpha)t_1
        exp_alpha_t_1 = pi_o[..., 1] / exp_alpha_exp_d_1

        # exp(alpha)t_2
        exp_alpha_t_2 = (pi_o[..., 2] -
                         exp_alpha_t_1 * exp_alpha_s_12) / exp_alpha_exp_d_2

        # exp(alpha)t_3
        exp_alpha_t_3 = (pi_o[..., 3] - exp_alpha_t_1 * exp_alpha_s_13 -
                         exp_alpha_t_2 * exp_alpha_s_23) / exp_alpha_exp_d_3

        # exp(alpha)
        # pylint: disable=E1103
        exp_alpha = torch.sqrt(pi_o[..., 0] - exp_alpha_t_1 ** 2 -
                               exp_alpha_t_2 ** 2 -
                               exp_alpha_t_3 ** 2).unsqueeze(-1)

        alpha = torch.log(exp_alpha)
        d_vector = torch.log(
            torch.stack(
                (exp_alpha_exp_d_1, exp_alpha_exp_d_2, exp_alpha_exp_d_3), -1) /
            exp_alpha)
        s_and_t = torch.stack(
            (exp_alpha_s_12, exp_alpha_s_23, exp_alpha_s_13, exp_alpha_t_1,
             exp_alpha_t_2, exp_alpha_t_3), -1) / exp_alpha
        return torch.cat((alpha, d_vector, s_and_t), -1)

    @staticmethod
    def pi_o_to_pi_cm(pi_o: Tensor) -> Tensor:
        """Converts batch of ``pi_o`` parameters to ``pi_cm`` parameters using
        the parallel axis theorem.

        Args:
            pi_o: ``(*, 10)`` ``pi_o``-type parameterization.

        Returns:
            pi_cm: ``(*, 10)`` ``pi_cm``-type parameterization.
        """
        # Expand in case tensor starts as shape (10,).
        pi_o = pi_o.reshape(-1, 10)

        # Split ``pi_o`` object into mass, CoM offset, and inertias wrt origin.
        mass = pi_o[..., 0].reshape(-1, 1)
        p_BoBcm_B = pi_o[..., 1:4] / mass
        I_BBo_B = pi_o[..., 4:]

        # Use parallel axis theorem to compute inertia matrix wrt CoM.
        inertia_mat = inertia_matrix_from_vector(I_BBo_B)
        new_inertia_mat = parallel_axis_theorem(inertia_mat, mass, p_BoBcm_B,
                                                from_com_to_other=False)
        I_BBcm_B = inertia_vector_from_matrix(new_inertia_mat)

        return torch.hstack((mass, p_BoBcm_B*mass, I_BBcm_B)).reshape(-1, 10)

    @staticmethod
    def pi_cm_to_pi_o(pi_cm: Tensor) -> Tensor:
        """Converts batch of ``pi_cm`` parameters to ``pi_o`` parameters using
        the parallel axis theorem.

        Args:
            pi_cm: ``(*, 10)`` ``pi_cm``-type parameterization.

        Returns:
            pi_o: ``(*, 10)`` ``pi_o``-type parameterization.
        """
        # Expand in case tensor starts as shape (10,).
        pi_cm = pi_cm.reshape(-1, 10)

        # Split ``pi_cm`` object into mass, CoM offset, and inertias wrt CoM.
        mass = pi_cm[..., 0].reshape(-1, 1)
        p_BoBcm_B = pi_cm[..., 1:4] / mass
        I_BBcm_B = pi_cm[..., 4:]

        # Use parallel axis theorem to compute inertia matrix wrt origin.
        inertia_mat = inertia_matrix_from_vector(I_BBcm_B)
        new_inertia_mat = parallel_axis_theorem(inertia_mat, mass, p_BoBcm_B,
                                                from_com_to_other=True)
        I_BBo_B = inertia_vector_from_matrix(new_inertia_mat)

        return torch.hstack((mass, p_BoBcm_B*mass, I_BBo_B)).reshape(-1, 10)

    @staticmethod
    def theta_to_pi_cm(theta: Tensor) -> Tensor:
        """Passthrough chain of :py:meth:`theta_to_pi_o` and
        :py:meth:`pi_o_to_pi_cm`."""
        pi_o = InertialParameterConverter.theta_to_pi_o(theta)
        return InertialParameterConverter.pi_o_to_pi_cm(pi_o)

    @staticmethod
    def pi_cm_to_theta(pi_cm: Tensor) -> Tensor:
        """Passthrough chain of :py:meth:`pi_cm_to_pi_o` and
        :py:meth:`pi_o_to_theta`."""
        pi_o = InertialParameterConverter.pi_cm_to_pi_o(pi_cm)
        return InertialParameterConverter.pi_o_to_theta(pi_o)

    @staticmethod
    def pi_cm_to_drake_spatial_inertia(pi_cm: Tensor) -> Tensor:
        """Converts batch of ``pi-cm`` parameters to ``drake_spatial_inertia``
        parameters."""
        # pylint: disable=E1103
        return torch.cat((pi_cm[..., 0:1], pi_cm[..., 1:] / pi_cm[..., 0:1]),
                         dim=-1)

    @staticmethod
    def pi_cm_to_urdf(pi_cm: Tensor) -> Tuple[str, str, List[str]]:
        """Converts a single ``(10,)`` ``pi_cm`` vector into the ``urdf`` string
        format."""
        assert len(pi_cm.shape) == 1
        mass = str(pi_cm[0].item())
        p_BoBcm_B = ' '.join(
            [str((coordinate / pi_cm[0]).item()) for coordinate in pi_cm[1:4]])
        I_BBcm_B = [str(inertia_element.item()) for inertia_element in pi_cm[4:]]

        return mass, p_BoBcm_B, I_BBcm_B

    @staticmethod
    def drake_to_pi_cm(spatial_inertia: DrakeSpatialInertia) -> Tensor:
        """Extracts a ``pi-cm`` parameterization from a Drake
        :py:attr:`~pydrake.multibody.tree.SpatialInertia` object."""
        mass = number_to_float(spatial_inertia.get_mass())
        p_BoBcm_B = spatial_inertia.get_com()
        I_BBcm_B = spatial_inertia.Shift(p_BoBcm_B).CalcRotationalInertia()

        mass_list = [
            mass * number_to_float(coordinate) for coordinate in p_BoBcm_B
        ]

        inertia_list = [
            number_to_float(I_BBcm_B[index[0], index[1]])
            for index in INERTIA_INDICES
        ]
        pi_cm = Tensor([mass] + mass_list + inertia_list)
        return pi_cm

    @staticmethod
    def drake_to_pi_o(spatial_inertia: DrakeSpatialInertia) -> Tensor:
        """Extracts a ``pi-o`` parameterization from a Drake
        :py:attr:`~pydrake.multibody.tree.SpatialInertia` object."""
        mass = number_to_float(spatial_inertia.get_mass())
        p_BoBcm_B = spatial_inertia.get_com()
        I_BBo_B = spatial_inertia.CalcRotationalInertia()

        mass_list = [
            mass * number_to_float(coordinate) for coordinate in p_BoBcm_B
        ]

        inertia_list = [
            number_to_float(I_BBo_B[index[0], index[1]])
            for index in INERTIA_INDICES
        ]
        pi_o = Tensor([mass] + mass_list + inertia_list)
        return pi_o

    @staticmethod
    def drake_to_theta(spatial_inertia: DrakeSpatialInertia) -> Tensor:
        """Passthrough chain of :py:meth:`drake_to_pi_o` and
        :py:meth:`pi_o_to_theta`."""
        pi_o = InertialParameterConverter.drake_to_pi_o(spatial_inertia)
        return InertialParameterConverter.pi_o_to_theta(pi_o)

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
