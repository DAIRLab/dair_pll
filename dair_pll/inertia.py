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

    * ``pi`` is an intuitive formatting of these 10 Dof as a standard vector in
      :math:`\mathbb{R}^{10}` as::

        [m, m * p_x, m * p_y, m * p_z, I_xx, I_yy, I_zz, I_xy, I_xz, I_yz]

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
        SpatialInertia.CalcRotationalInertia() == I_BBcm_B

    * ``theta`` is a format designed for underlying smooth, unconstrained,
      and non-degenerate parameterization of rigid body inertia. For a body,
      any value in :math:`\mathbb{R}^{10}` for ``theta`` can be mapped to a
      valid and non-degenerate set of inertial terms as follows::

        theta == [alpha, d_1, d_2, d_3, s_12, s_23, s_13, t_1, t_2, t_3]
        s == [s_12, s_23, s_13]
        t == [t_1, t_2, t_3]
        pi == [
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
from dair_pll.tensor_utils import deal

INERTIA_INDICES = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
INERTIA_SCALARS = ["I_xx", "I_yy", "I_zz", "I_xy", "I_xz", "I_yz"]
AXES = ["x", "y", "z"]


def number_to_float(number: Any) -> float:
    """Converts a number to float via intermediate string representation."""
    return float(str(number))


class InertialParameterConverter:
    """Utility class for transforming between inertial parameterizations."""

    @staticmethod
    def theta_to_pi(theta: Tensor) -> Tensor:
        """Converts batch of ``theta`` parameters to ``pi`` parameters.

        Args:
            theta: ``(*, 10)`` ``theta``-type parameterization.

        Returns:
            ``(*, 10)`` ``pi``-type parameterization.
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
    def pi_to_theta(pi: Tensor) -> Tensor:
        """Converts batch of ``pi`` parameters to ``theta`` parameters.

        Implements hand-derived local inverse of standard mapping from Rucker
        and Wensing.

        This function inverts :py:meth:``theta_to_pi`` for
        valid ``pi``.

        Args:
            pi: ``(*, 10)`` ``pi``-type parameterization.

        Returns:
            ``(*, 10)`` ``theta``-type parameterization.
        """

        # exp(alpha)exp(d_1)
        # pylint: disable=E1103
        exp_alpha_exp_d_1 = torch.sqrt(0.5 *
                                       (pi[..., 5] + pi[..., 6] - pi[..., 4]))

        # exp(alpha)s_12
        exp_alpha_s_12 = -pi[..., 7] / exp_alpha_exp_d_1

        # exp(alpha)s_13
        exp_alpha_s_13 = -pi[..., 8] / exp_alpha_exp_d_1

        # exp(alpha)exp(d_2)
        # pylint: disable=E1103
        exp_alpha_exp_d_2 = torch.sqrt(pi[..., 6] - exp_alpha_exp_d_1 ** 2 -
                                       exp_alpha_s_12 ** 2)

        # exp(alpha)s_23
        exp_alpha_s_23 = (-pi[..., 9] -
                          exp_alpha_s_12 * exp_alpha_s_13) / exp_alpha_exp_d_2

        # exp(alpha)exp(d3)
        # pylint: disable=E1103
        exp_alpha_exp_d_3 = torch.sqrt(pi[..., 5] - exp_alpha_exp_d_1 ** 2 -
                                       exp_alpha_s_13 ** 2 - exp_alpha_s_23 ** 2)

        # exp(alpha)t_1
        exp_alpha_t_1 = pi[..., 1] / exp_alpha_exp_d_1

        # exp(alpha)t_2
        exp_alpha_t_2 = (pi[..., 2] -
                         exp_alpha_t_1 * exp_alpha_s_12) / exp_alpha_exp_d_2

        # exp(alpha)t_3
        exp_alpha_t_3 = (pi[..., 3] - exp_alpha_t_1 * exp_alpha_s_13 -
                         exp_alpha_t_2 * exp_alpha_s_23) / exp_alpha_exp_d_3

        # exp(alpha)
        # pylint: disable=E1103
        exp_alpha = torch.sqrt(pi[..., 0] - exp_alpha_t_1 ** 2 -
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
    def pi_to_drake_spatial_inertia(pi: Tensor) -> Tensor:
        """Converts batch of ``pi`` parameters to ``drake_spatial_inertia``
        parameters."""
        # pylint: disable=E1103
        return torch.cat((pi[..., 0:1], pi[..., 1:] / pi[..., 0:1]), dim=-1)

    @staticmethod
    def pi_to_urdf(pi: Tensor) -> Tuple[str, str, List[str]]:
        """Converts a single ``(10,)`` ``pi`` vector into the ``urdf`` string
        format."""
        assert len(pi.shape) == 1
        mass = str(pi[0].item())
        p_BoBcm_B = ' '.join(
            [str((coordinate / pi[0]).item()) for coordinate in pi[1:4]])
        I_BBcm_B = [str(inertia_element.item()) for inertia_element in pi[4:]]

        return mass, p_BoBcm_B, I_BBcm_B

    @staticmethod
    def drake_to_pi(spatial_inertia: DrakeSpatialInertia) -> Tensor:
        """Extracts a ``pi`` parameterization from a Drake
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
        pi = Tensor([mass] + mass_list + inertia_list)
        return pi

    @staticmethod
    def drake_to_theta(spatial_inertia: DrakeSpatialInertia) -> Tensor:
        """Passthrough chain of :py:meth:`drake_to_pi` and
        :py:meth:`pi_to_theta`."""
        pi = InertialParameterConverter.drake_to_pi(spatial_inertia)
        return InertialParameterConverter.pi_to_theta(pi)

    @staticmethod
    def pi_to_scalars(pi: Tensor) -> Dict[str, float]:
        """Converts ``pi`` parameterization to ``scalars`` dictionary."""
        mass = pi[0]
        p_BoBcm_B = pi[1:4] / mass
        I_BBcm_B = pi[4:]
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
