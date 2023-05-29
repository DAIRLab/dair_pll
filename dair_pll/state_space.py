r"""Classes and utilities for operation on Lie group/algebra state spaces.

This module implements a :py:class:`StateSpace` abstract type which defines
fundamental operations on states in spaces that are not necessarily
Euclidean. In general, we assume configurations spaces are Lie groups,
which together with their associated Lie algebra form a state space. We
implement associated operations on these spaces.

Of note is the :py:class:`FloatingBaseSpace`, with configurations in the
Cartesian product of :math:`SE(3)` and :math:`\mathbb{R}^m`. This space receives
the  particular implementation of representing the :math:`SO(3)`
configuration as a quaternion and the Lie algebra velocity/tangent space as
the body-axes angular velocity / rotation vector.

This is also the place where batching dimensions are defined for states. By
convention, the state element index is always the last dimension of the
tensor, and when states are batched in time, time is the second-to-last index.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Dict, cast

import torch
from torch import Tensor

from dair_pll import quaternion

N_QUAT = 4
N_ANG_VEL = 3
N_COM = 3

ComparisonCallable = Callable[[Tensor, Tensor], Tensor]
ComparisonDict = Dict[str, ComparisonCallable]
# pylint: disable=E1103
Size = torch.Size


def partial_sum_batch(summands: Tensor, keep_batch: bool = False) -> Tensor:
    """Sums over a batch, possibly keeping the first batching dimension.

    Args:
        summands: ``(b_1, ..., b_j, n_1, ..., n_k)`` tensor batch of tensors
        keep_batch: whether to keep the first batching dimension

    Returns:
        sum of x as scalar tensor, or ``(b_1,)`` tensor if
        ``keep_batch == True``.
    """
    if keep_batch:
        while len(summands.shape) > 1:
            summands = summands.sum(dim=-1)
        return summands
    return summands.sum()


class StateSpace(ABC):
    r"""Mathematical model of a state space.

    Each state space is modeled as the Cartesian product of a connected Lie
    group :math:`G` and its associated Lie algebra :math:`\mathfrak g` (
    equivalently up to diffeomorphism, the tangent bundle :math:`TG`).
    The Lie group element may be given non-minimal coordinates,
    e.g. representing SO(3) with quaternions. As :math:`\mathfrak g`
    is a vector space, :math:`G \times \mathfrak g` itself is also a Lie group.

    The following assumptions about the group :math:`G` and algebra g are made:

        * The Lie group exponential map :math:`\exp: \mathfrak g \to G`
          is surjective/onto, such that a left inverse
          :math:`\log: G \to \mathfrak g` can be defined, i.e.
          :math:`\exp(\log(g)) = g`.
        * The Lie group exponential map coincides with the underlying
          manifold's Riemannian geometric exponential map, such that the
          geodesic distance from :math:`g_1` to :math:`g_1` is
          :math:`|\log(g_2 \cdot g_1^{-1})|`

    These conditions are met if and only if :math:`G` is the Cartesian
    product of a compact group and an Abelian group :cite:p:`Milnor1976` -- For
    example, :math:`SO(3)\times\mathbb{R}^n`.

    For each concrete class inheriting from :py:class:`StateState`,
    a few fundamental mathematical operators associated with Lie groups must
    be defined on these coordinates. :py:class:`StateState` defines several
    other group operations from these units.
    """
    n_q: int
    n_v: int
    n_x: int
    comparisons: ComparisonDict

    def __init__(self, n_q: int, n_v: int) -> None:
        """
        Args:
            n_q: number of Lie group (configuration) coordinates
              (:math:`>= 0`)
            n_v: number of Lie algebra (velocity) coordinates (:math:`>= 0`)
        """
        assert n_q >= 0
        assert n_v >= 0
        super().__init__()
        self.n_q = n_q
        self.n_v = n_v
        self.n_x = n_q + n_v
        self.comparisons = {}

    @abstractmethod
    def configuration_difference(self, q_1: Tensor, q_2: Tensor) -> Tensor:
        r"""Returns the relative transformation between ``q_1`` and ``q_2``.

        Specifically, as :math:`G` is a Lie group, it has a well-defined inverse
        operator. This function returns :math:`dq = \log(q_2 \cdot q_1^{-1})`,
        i.e. the Lie algebra element such that :math:`q_1 \exp(dq) = q_2`.

        This method has a corresponding "inverse" function
        :py:meth:`exponential`.

        Args:
            q_1: ``(*, n_q)`` "starting" configuration, element(s) of Lie
              group :math:`G`.
            q_2: ``(*, n_q)`` "ending" configuration, element(s) of Lie group
              :math:`G`.

        Returns:
            ``(*, n_v)`` element of Lie algebra g defining the transformation
              from ``q_1`` to ``q_2``
        """

    @abstractmethod
    def exponential(self, q: Tensor, dq: Tensor) -> Tensor:
        """Evolves ``q`` along the Lie group G in the direction ``dq``.

        This function implements the inverse of
        :py:meth:`configuration_difference` by returning q * exp(dq).

        Args:
            q: ``(*, n_q)`` "starting" configuration, element(s) of Lie group G
            dq: ``(*, n_v)`` perturbation, element(s) of Lie algebra g
        Returns:
            ``(*, n_q)`` group product of q and exp(dq)
        """

    @abstractmethod
    def project_configuration(self, q: Tensor) -> Tensor:
        """Projects a tensor of size ``(*, n_q)`` onto the Lie group G.

        This function is used, mostly for numerical stability, to ensure a
        ``(*, n_q)`` tensor corresponds to Lie group elements. While not
        necessarily a Euclidean projection, this function should be:

            * The identity on G, i.e. ``q = projection_configuration(q)``
            * Continuous
            * (Piecewise) differentiable near G

        Args:
            q: ``(*, n_q)`` vectors to project onto G.

        Returns:
            ``(*, n_q)`` projection of ``q`` onto G.
        """

    @abstractmethod
    def zero_state(self) -> Tensor:
        """Identity element of the Lie group G.

        Entitled "zero state" as the group operation is typically thought of
        as addition.

        Returns:
            ``(n_x,)`` tensor group identity
        """

    def q(self, x: Tensor) -> Tensor:
        """Selects configuration indices from state(s) ``x``"""
        assert x.shape[-1] == self.n_x
        return x[..., :self.n_q]

    def v(self, x: Tensor) -> Tensor:
        """Selects velocity indices from state(s) ``x``"""
        assert x.shape[-1] == self.n_x
        return x[..., self.n_q:]

    def q_v(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Separates state(s) ``x`` into configuration and velocity"""
        assert x.shape[-1] == self.n_x
        return self.q(x), self.v(x)

    def x(self, q: Tensor, v: Tensor) -> Tensor:
        """Concatenates configuration ``q`` and velocity ``v`` into a state"""
        assert q.shape[-1] == self.n_q
        assert v.shape[-1] == self.n_v

        # pylint: disable=E1103
        return torch.cat((q, v), dim=-1)

    def config_square_error(self,
                            q_1: Tensor,
                            q_2: Tensor,
                            keep_batch: bool = False) -> Tensor:
        r"""Returns squared distance between two Lie group
        elements/configurations.

        Interprets an :math:`l_2`-like error between two configurations as the
        square of the geodesic distance between them. This is simply equal to
        :math:`|\log(q_2 \mathrm{inv}(q_1))|^2` under the assumptions about G.

        Args:
            q_1: ``(b_1, ..., b_k, n_q)`` "starting" configuration
            q_2: ``(b_1, ..., b_k, n_q)`` "ending" configuration
            keep_batch: whether to keep the outermost batch

        Returns:
            ``(b_1,)`` or scalar tensor of squared geodesic distances
        """
        assert q_1.shape[-1] == self.n_q
        assert q_2.shape[-1] == self.n_q
        return partial_sum_batch(
            self.configuration_difference(q_1, q_2)**2, keep_batch)

    def velocity_square_error(self,
                              v_1: Tensor,
                              v_2: Tensor,
                              keep_batch: bool = False) -> Tensor:
        """Returns squared distance between two Lie algebra
        elements/velocities.

        As the Lie algebra is a vector space, the squared error is
        interpreted as the geodesic/Euclidean distance

        Args:
            v_1: ``(b_1, ..., b_k, n_v)`` "starting" velocity
            v_2: ``(b_1, ..., b_k, n_v)`` "ending" velocity
            keep_batch: whether to keep the outermost batch

        Returns:
            ``(b_1,)`` or scalar tensor of squared geodesic distances.
        """
        assert v_1.shape[-1] == self.n_v
        assert v_2.shape[-1] == self.n_v
        return partial_sum_batch((v_2 - v_1)**2, keep_batch)

    def state_square_error(self,
                           x_1: Tensor,
                           x_2: Tensor,
                           keep_batch: bool = False) -> Tensor:
        """Returns squared distance between two states, which are in the
        cartesian product G x g.

        As g is a vector space, it is Abelian, and thus G x g is the product
        of a compact group and an Abelian group. We can then define the
        geodesic distance as::

            dist(x_1, x_2)^2 == dist(q(x_1), q(x_2))^2 + dist(v(x_1), v(x_2))^2

        Args:
            x_1: ``(b_1, ..., b_k, n_x)`` "starting" state
            x_2: ``(b_1, ..., b_k, n_x)`` "ending" state
            keep_batch: whether to keep the outermost batch

        Returns:
            ``(b_1,)`` or scalar tensor of squared geodesic distances
        """
        assert x_1.shape[-1] == self.n_x
        assert x_2.shape[-1] == self.n_x
        q_1, v_1 = self.q_v(x_1)
        q_2, v_2 = self.q_v(x_2)
        return self.config_square_error(
            q_1, q_2, keep_batch) + self.velocity_square_error(
                v_1, v_2, keep_batch)

    def auxiliary_comparisons(
            self) -> Dict[str, Callable[[Tensor, Tensor], Tensor]]:
        """Any additional useful comparisons between pairs of states"""
        return self.comparisons

    def finite_difference(self, q: Tensor, q_plus: Tensor, dt: float) -> Tensor:
        """Rate of change of configuration

        Interprets the rate of change of ``q`` as an element of the Lie
        algebra ``v``, such that q_plus == q * exp(v * dt).

        :py:meth:`finite_difference` has a corresponding "inverse" function
        :py:meth:`euler_step`.

        Args:
            q: ``(*, n_q)`` "starting" configuration, element(s) of Lie group G
            q_plus: ``(*, n_q)`` "ending" configuration, element(s) of Lie group G
            dt: time difference in [s]

        Returns:
            ``(*, n_v)`` finite-difference velocity, element(s) of Lie algebra g
        """
        assert q.shape[-1] == self.n_q
        assert q_plus.shape[-1] == self.n_q
        return self.configuration_difference(q, q_plus) / dt

    def euler_step(self, q: Tensor, v: Tensor, dt: float) -> Tensor:
        """Integrates ``q`` forward in time given derivative ``v``.

        Implements the inverse of :py:meth:`finite_difference` by returning
        q * exp(v * dt), a geodesic forward Euler step.

        Args:
            q: ``(*, n_q)`` "starting" configuration, element(s) of Lie group G
            v: ``(*, n_v)`` "starting" velocity, element(s) of Lie algebra g
            dt: time difference in [s]

        Returns:
            ``(*, n_q)`` configuration after Euler step.
        """
        assert q.shape[-1] == self.n_q
        assert v.shape[-1] == self.n_v
        return self.exponential(q, v * dt)

    def state_difference(self, x_1: Tensor, x_2: Tensor) -> Tensor:
        """Returns the relative transformation between ``x_1`` and ``x_2``

        As G x g is a Lie group, we can interpret the difference between two
        states via its corresponding Lie algebra, just as in
        :py:meth:`configuration_difference`, as log(x_2 / x_1).

        :py:meth:`state_difference` has a corresponding "inverse" function
        :py:meth:`shift_state`.

        Args:
            x_1: ``(*, n_x)`` "starting" state, element(s) of Lie group G x g
            x_2: ``(*, n_x)`` "ending" state, element(s) of Lie group G x g

        Returns:
            ``(*, n_x)`` element of Lie algebra g x R^n_v defining the
            transformation from ``x_1`` to ``x_2``
        """
        assert x_1.shape[-1] == self.n_x
        assert x_2.shape[-1] == self.n_x
        q_1, v_1 = self.q_v(x_1)
        q_2, v_2 = self.q_v(x_2)
        dq = self.configuration_difference(q_1, q_2)
        dv = v_2 - v_1

        # pylint: disable=E1103
        return torch.cat((dq, dv), dim=-1)

    def shift_state(self, x: Tensor, dx: Tensor) -> Tensor:
        """Evolves ``x`` along the Lie group G in the direction ``dx``.

        This function implements the inverse of :py:meth:`state_difference`
        by returning q * exp(dq).

        Args:
            x: ``(*, n_x)`` "starting" state, element(s) of Lie group G x g
            dx: ``(*, 2 * n_v)`` perturbation, element(s) of Lie algebra g x R^n_v
        Returns:
            ``(*, n_q)`` group product of q and exp(dq).
        """
        assert x.shape[-1] == self.n_x
        assert dx.shape[-1] == (2 * self.n_v)
        q, v = self.q_v(x)

        dq = dx[..., :self.n_v]
        dv = dx[..., self.n_v:]

        q_new = self.exponential(q, dq)
        v_new = v + dv
        return self.x(q_new, v_new)

    def project_state(self, x: Tensor) -> Tensor:
        """Projects a tensor of size (\*, n_x) onto the state space G x g.

        This function has the same basic requirements as
        :py:meth:`project_configuration` translated to the lie group G x g.

        Args:
            x: ``(*, n_x)`` vectors to project onto G x g.

        Returns:
            ``(*, n_x)`` tensor, projection of ``x`` onto G x g.
        """
        assert x.shape[-1] == self.n_x
        return self.x(self.project_configuration(self.q(x)), self.v(x))

    def project_derivative(self, x: Tensor, dt: float) -> Tensor:
        """Changes velocity sequence in ``x`` to a finite difference.

        Extracts configurations q_t from trajectory(ies) and replaces
        velocities v_i with ``finite_difference(q_{i-1}, q_i, dt)``.

        Args:
            x: ``(*, T, n_x)`` trajectories
            dt: time-step

        Returns:
            ``(*, T, n_x)`` trajectories with finite-difference velocities.
        """
        assert x.shape[-1] == self.n_x
        assert x.dim() >= 2  # must have time indexing
        assert x.shape[-2] > 1  # must have multiple time steps
        q = self.q(x)
        q_pre = q[..., :(-1), :]
        q_plus = q[..., 1:, :]
        v_plus = self.finite_difference(q_pre, q_plus, dt)
        return self.x(q_plus, v_plus)


class FloatingBaseSpace(StateSpace):
    """State space with configurations in SE(3) x R^n_joints.

    Called :py:class:`FloatingBaseSpace` as it is the state space of an open
    kinematic chain with a free-floating base body.

    Coordinates for SO(3) are unit quaternions, with remaining states
    represented as R^{3 + n_joints}.
    """

    def __init__(self, n_joints: int) -> None:
        """Inits `:py:class:`FloatingBaseSpace` of prescribed size.

        The floating base has configurations in SE(3), with 4 quaternion + 3
        world-axes position configuration coordinates and 3 body-axes angular
        velocity + 3 world-axes linear velocity. Each joint is represented as a
        single real number.

        Args:
            n_joints: number of joints in chain (>= 0)
        """
        assert n_joints >= 0
        super().__init__(7 + n_joints, 6 + n_joints)
        self.comparisons.update({
            'rot_err': self.quaternion_error,
            'pos_err': self.base_error,
        })

    def quat(self, q_or_x: Tensor) -> Tensor:
        """select quaternion elements from configuration/state"""
        assert q_or_x.shape[-1] == self.n_q or q_or_x.shape[-1] == self.n_x
        return q_or_x[..., :N_QUAT]

    def base(self, q_or_x: Tensor) -> Tensor:
        """select floating base position elements from configuration/state"""
        assert q_or_x.shape[-1] == self.n_q or q_or_x.shape[-1] == self.n_x
        return q_or_x[..., N_QUAT:(N_QUAT + N_COM)]

    def configuration_difference(self, q_1: Tensor, q_2: Tensor) -> Tensor:
        """Implements configuration offset for a floating-base rigid chain.

        exp() map of SO(3) corresponds to the space of rotation
        vectors, or equivalently the matrix group so(3); therefore, the first 3
        elements of the return value are body-axes rotation vectors.

        Args:
            q_1: ``(*, n_q)`` "starting" configuration in SE(3) x R^n_joints
            q_2: ``(*, n_q)`` "ending" configuration, SE(3) x R^n_joints

        Returns:
            ``(*, n_v)`` body-axes rotation vector, world-axes linear
            displacement, and joint offsets.
        """
        assert q_1.shape[-1] == self.n_q
        assert q_2.shape[-1] == self.n_q
        quat1 = self.quat(q_1)
        quat2 = self.quat(q_2)
        linear_shift = q_2[..., N_QUAT:] - q_1[..., N_QUAT:]
        quat_shift = quaternion.multiply(quaternion.inverse(quat1), quat2)
        rot = quaternion.log(quat_shift)

        # pylint: disable=E1103
        return torch.cat((rot, linear_shift), dim=-1)

    def exponential(self, q: Tensor, dq: Tensor) -> Tensor:
        """Implements exponential perturbation for a floating-base rigid chain.

        This function implements the inverse of :py:meth:`configuration_difference`
        by rotating ``quat(q)`` around the body-axis rotation vector in
        ``dq``, and adding a linear offset to the remaining coordinates.

        Args:
            q: ``(*, n_q)`` "starting" configuration in SE(3) x R^n_joints
            dq: ``(*, n_v)`` perturbation in se(3) x R^n_joints
        Returns:
            ``(*, n_q)`` perturbed quaternion, world-axes floating base origin
        """
        assert q.shape[-1] == self.n_q
        assert dq.shape[-1] == self.n_v
        linear_plus = q[..., N_QUAT:] + dq[..., N_ANG_VEL:]
        delta_quat = quaternion.exp(dq[..., :N_ANG_VEL])
        quat_plus = quaternion.multiply(q[..., :N_QUAT], delta_quat)

        # pylint: disable=E1103
        return torch.cat((quat_plus, linear_plus), dim=-1)

    def project_configuration(self, q: Tensor) -> Tensor:
        """Implements projection onto the floating-base rigid chain
        configuration space.

        This function projects a ``(*, n_q)`` tensor onto SE(3) x R^n_joints by
        simply normalizing the quaternion elements.

        Args:
            q: ``(*, n_q)`` vectors to project onto SE(3) x R^n_joints.

        Returns:
            ``(*, n_q)`` tensor, projection of ``q`` onto SE(3) x R^n_joints.
        """
        assert q.shape[-1] == self.n_q
        quats = q[..., :N_QUAT] / torch.linalg.norm(q[..., :N_QUAT], dim=-1)

        # pylint: disable=E1103
        return torch.cat((quats, q[..., N_QUAT:]), dim=-1)

    def zero_state(self) -> Tensor:
        """Identity element of SE(3) x R^n_joints.

        Returns:
            Concatenation of identity quaternion [1, 0, 0, 0] with
            ``(n_joints + 3)`` zeros.
        """
        # pylint: disable=E1103
        zero = torch.zeros((self.n_x,))
        zero[0] = 1.
        return zero

    def quaternion_error(self, x_1: Tensor, x_2: Tensor) -> Tensor:
        """Auxiliary comparison that returns floating base orientation geodesic
        distance.

        Returns a scalar comparison of two floating base rigid chain states
        by the angle of rotation between their base orientations.

        Args:
            x_1: ``(*, n_x)`` "starting" state
            x_2: ``(*, n_x)`` "ending" state

        Returns:
            scalar tensor, average angle of rotation in batch.

        Todo:
            Properly handle multiple batching dimensions.
        """
        assert x_1.shape[-1] == self.n_x
        assert x_2.shape[-1] == self.n_x
        assert len(x_1.shape) == 2  # hack for now
        quat1 = self.quat(x_1)
        quat2 = self.quat(x_2)
        quat_shift = quaternion.multiply(quaternion.inverse(quat1), quat2)
        rot = quaternion.log(quat_shift)

        # pylint: disable=E1103
        return rot.norm(dim=-1).sum() / x_1.shape[0]

    def base_error(self, x_1: Tensor, x_2: Tensor) -> Tensor:
        """Auxiliary comparison that returns floating base translation geodesic
        distance.

        Returns a scalar comparison of two floating base rigid chain states
        by the Euclidean between their bases.

        Args:
            x_1: ``(*, n_x)`` "starting" state
            x_2: ``(*, n_x)`` "ending" state

        Returns:
            scalar tensor, average translation in batch.

        Todo:
            Properly handle multiple batching dimensions.
        """
        assert x_1.shape[-1] == self.n_x
        assert x_2.shape[-1] == self.n_x
        assert len(x_1.shape) == 2
        base1 = self.base(x_1)
        base2 = self.base(x_2)
        pos = base1 - base2

        # pylint: disable=E1103
        return pos.norm(dim=-1).sum() / x_1.shape[0]


class FixedBaseSpace(StateSpace):
    """State space with configurations in R^n_joints.

    Called :py:class:`FixedBaseSpace` as it is the state space of an open
    kinematic chain with fixed base body.

    As the Lie group R^n_joints is equivalent to its own algebra, the state
    space is simply R^{2 * n_joints}, and the group operation coincides with
    addition on this vector space. Thus::

        n_q == n_v == n_x/2
    """

    def __init__(self, n_joints: int) -> None:
        """Inits :py:class:`FixedBaseSpace` of prescribed size.

        Args:
            n_joints: number of joints in chain (>= 0)
        """
        assert n_joints >= 0
        super().__init__(n_joints, n_joints)

    def configuration_difference(self, q_1: Tensor, q_2: Tensor) -> Tensor:
        """Implements configuration offset for a fixed-base rigid chain.

        In R^n_joints, this is simply vector subtraction.

        Args:
            q_1: ``(*, n_q)`` "starting" configuration in R^n_joints
            q_2: ``(*, n_q)`` "ending" configuration in R^n_joints

        Returns:
            (\*, n_v) difference of configurations
        """
        assert q_1.shape[-1] == self.n_q
        assert q_2.shape[-1] == self.n_q
        return q_2 - q_1

    def exponential(self, q: Tensor, dq: Tensor) -> Tensor:
        """Implements exponential perturbation for a fixed-base rigid chain.

        In R^n_joints, this is simply vector addition.

        Args:
            q: ``(*, n_q)`` "starting" configuration in R^n_joints
            dq: ``(*, n_v)`` perturbation in R^n_joints
        Returns:
            ``(*, n_q)`` perturbed configuration
        """
        assert q.shape[-1] == self.n_q
        assert dq.shape[-1] == self.n_v
        return q + dq

    def project_configuration(self, q: Tensor) -> Tensor:
        """Implements projection onto the fixed-base rigid chain
        configuration space.

        In R^n_joints, this is simply the identity function.

        Args:
            q: ``(*, n_q)`` vectors in R^n_joints.

        Returns:
            ``(*, n_q)`` tensor, ``q``.
        """
        assert q.shape[-1] == self.n_q
        return q

    def zero_state(self) -> Tensor:
        """Zero element of R^n_x"""

        # pylint: disable=E1103
        return torch.zeros((self.n_x,))


class ProductSpace(StateSpace):
    """State space constructed as the Cartesian product of subspaces.

    The product space conforms with the required properties of our Lie group
    as long as each constituent subspace does as well."""

    def __init__(self, spaces: List[StateSpace]) -> None:
        """Inits :py:class:`ProductSpace` from given factors.

        The coordinates of each space in ``spaces`` are concatenated to
        construct the product space's coordinates, and similar for
        the velocities."""
        n_qs = [space.n_q for space in spaces]
        n_vs = [space.n_v for space in spaces]
        n_xs = [space.n_x for space in spaces]
        n_q = sum(n_qs)
        n_v = sum(n_vs)

        super().__init__(n_q, n_v)
        # pylint: disable=E1103
        self.q_splits = torch.cumsum(torch.tensor(n_qs), 0)[:-1]
        self.v_splits = torch.cumsum(torch.tensor(n_vs), 0)[:-1]
        self.x_splits = torch.cumsum(torch.tensor(n_xs), 0)[:-1]
        self.spaces = spaces
        for i, space in enumerate(spaces):
            self.comparisons.update({
                f'{name}_{i}': self.indexed_state_comparison(i, comparison)
                for name, comparison in space.comparisons.items()
            })

    def indexed_state_comparison(
            self, space_num: int,
            comparison: ComparisonCallable) -> ComparisonCallable:
        """Returns a state comparison function for the given subspace.

        Args:
            space_num: index of subspace to compare.
            comparison: comparison function for subspace.
        """
        return lambda x_1, x_2: comparison(
            self.x_split(x_1)[space_num],
            self.x_split(x_2)[space_num])

    def q_split(self, q: Tensor) -> List[Tensor]:
        """Splits configuration into list of subspace configurations."""
        assert q.shape[-1] == self.n_q

        # pylint: disable=E1103
        return cast(list, torch.tensor_split(q, self.q_splits, -1))

    def v_split(self, v: Tensor) -> List[Tensor]:
        """Splits velocity into list of subspace velocities."""
        assert v.shape[-1] == self.n_v

        # pylint: disable=E1103
        return cast(list, torch.tensor_split(v, self.v_splits, -1))

    def x_split(self, x: Tensor) -> List[Tensor]:
        """Splits state into list of subspace states."""
        assert x.shape[-1] == self.n_x

        # pylint: disable=E1103
        return cast(list, torch.tensor_split(x, self.x_splits, -1))

    def configuration_difference(self, q_1: Tensor, q_2: Tensor) -> Tensor:
        """Constructs configuration difference as concatenation of subspace
        configuration differences."""
        assert q_1.shape[-1] == self.n_q
        assert q_2.shape[-1] == self.n_q
        diffs = [
            space.configuration_difference(q_1i, q_2i)
            for space, q_1i, q_2i in zip(self.spaces, self.q_split(q_1),
                                         self.q_split(q_2))
        ]
        # pylint: disable=E1103
        return torch.cat(diffs, dim=-1)

    def exponential(self, q: Tensor, dq: Tensor) -> Tensor:
        """Constructs perturbed configuration as concatenation of perturbed
        subspace configurations"""
        assert q.shape[-1] == self.n_q
        assert dq.shape[-1] == self.n_v
        exps = [
            space.exponential(qi, dqi) for space, qi, dqi in zip(
                self.spaces, self.q_split(q), self.v_split(dq))
        ]
        # pylint: disable=E1103
        return torch.cat(exps, dim=-1)

    def project_configuration(self, q: Tensor) -> Tensor:
        """Projects configuration onto Lie group by projecting each subspace's
        configuration onto its respective subgroup."""
        assert q.shape[-1] == self.n_q
        projections = [
            space.project_configuration(qi)
            for space, qi in zip(self.spaces, self.q_split(q))
        ]
        # pylint: disable=E1103
        return torch.cat(projections, dim=-1)

    def zero_state(self) -> Tensor:
        """Constructs zero state as concatenation of subspace zeros"""
        zeros = [space.zero_state() for space in self.spaces]

        # pylint: disable=E1103
        q = torch.cat(
            [space.q(zero) for space, zero in zip(self.spaces, zeros)], dim=-1)
        v = torch.cat(
            [space.v(zero) for space, zero in zip(self.spaces, zeros)], dim=-1)
        return torch.cat((q, v), dim=-1)


def centered_uniform(size: Size) -> Tensor:
    """Uniform distribution on zero-centered box [-1, 1]^size"""
    # pylint: disable=E1103
    return 2. * torch.rand(size) - 1.


class WhiteNoiser:
    r"""Helper class for adding artificial noise to state batches.

    Defines an interface for noise distortion of a batch of states. Noise is
    modeled as a zero-mean distribution on the Lie algebra of the state space,
    :math:`\mathbb{R}^{2 n_v}`. Note that this means that velocities receive
    noise independent to the configuration, and thus may break the
    finite-difference relationship in a trajectory."""
    space: StateSpace
    ranges: Tensor
    variance_factor: float

    def __init__(self,
                 space: StateSpace,
                 unit_noise: Callable[[Size], Tensor],
                 variance_factor: float = 1) -> None:
        """Inits a :py:class:`WhiteNoiser` of specified distribution.

        Args:
            space: State space upon which
            unit_noise: Callback, returns coordinate-independent noise of
              nominal unit size.
            variance_factor: Variance of a single coordinate's unit-scale noise.
        """
        super().__init__()
        self.space = space
        self.unit_noise = unit_noise
        self.variance_factor = variance_factor

    def noise(self,
              x: Tensor,
              ranges: Tensor,
              independent: bool = True) -> Tensor:
        """Adds noise to a given batch of states.

        Uses the ``unit_noise()`` to get. Optionally, adds identical
        distortion to each state in the batch, or i.i.d. noise to each state.

        Args:
            x: ``(*, space.n_x)`` batch of states to distort with noise.
            ranges: ``(2 * space.n_v,)`` multiplicative scale of noise.
            independent: whether to independently distort each state.

        Returns:
            ``(*, space.n_x)`` distorted batch of states.
        """
        dx_shape = x.shape[:-1] + (2 * self.space.n_v,)
        if independent:
            noise_shape = torch.Size(dx_shape)
        else:
            noise_shape = torch.Size((2 * self.space.n_v,))

        noise = torch.zeros(dx_shape)
        noise += self.unit_noise(noise_shape) * ranges
        return self.space.shift_state(x, noise)

    def covariance(self, ranges: Tensor) -> Tensor:
        """State covariance matrix associated with noise scale.

        Args:
            ranges: ``(2 * space.n_v,)`` multiplicative scale of noise.

        Returns:
            ``(2 * space.n_v, 2 * space.n_v)`` covariance matrix on state space
            Lie algebra.
        """
        return torch.diag(self.variance_factor * (ranges**2))


class UniformWhiteNoiser(WhiteNoiser):
    """Convenience :py:class:`WhiteNoiser` class for uniform noise."""

    def __init__(self, space: StateSpace) -> None:
        super().__init__(space, centered_uniform, 1. / 3.)


class GaussianWhiteNoiser(WhiteNoiser):
    """Convenience :py:class:`WhiteNoiser` class for Gaussian noise."""

    def __init__(self, space: StateSpace) -> None:
        super().__init__(space, torch.randn)


class StateSpaceSampler(ABC):
    """Abstract utility class for sampling on a state space."""
    space: StateSpace

    def __init__(self, space: StateSpace) -> None:
        """Inits :py:class:`StateSpaceSampler` on prescribed space.

        Args:
            space: State space of sampler.
        """
        super().__init__()
        self.space = space

    @abstractmethod
    def get_sample(self) -> Tensor:
        """Get sample from state distribution.

        Returns:
            (space.n_x,) state sample.
        """

    @abstractmethod
    def covariance(self) -> Tensor:
        r"""Returns covariance of state space distribution.

        Interprets the distribution in logarithmic coordinates (the Lie
        algebra of the state space), and returns a covariance matrix in
        :math:`\mathbb{R}^{2 n_v \times 2 n_v}`.

        Returns:
            (2 * space.n_v, 2 * space.n_v) distribution covariance.
        """


class ConstantSampler(StateSpaceSampler):
    """Convenience :py:class:`StateSpaceSampler` for returning constant
    state."""
    space: StateSpace
    x_0: Tensor

    def __init__(self, space: StateSpace, x_0: Tensor) -> None:
        """Inits :py:class:`ConstantSampler` with specified constant state.

        Args:
            space: Sampler's state space.
            x_0: ``(space.n_x,)`` singleton support of underlying probability
              distribution.
        """
        super().__init__(space)
        self.x_0 = x_0

    def get_sample(self) -> Tensor:
        """Returns copy of constant ``x_0``."""
        return self.x_0.clone()

    def covariance(self) -> Tensor:
        """Returns zero covariance."""
        return torch.zeros((2 * self.space.n_v, 2 * self.space.n_v))


class ZeroSampler(ConstantSampler):
    """Convenience :py:class:`ConstantSampler` for returning zero state."""

    def __init__(self, space: StateSpace) -> None:
        super().__init__(space, space.zero_state())


class CenteredSampler(StateSpaceSampler):
    """State space sampling distribution centered around specified state.

    Implemented by sampling the state, and perturbing it with specified white
    noise.
    """
    ranges: Tensor
    x_0: Tensor
    variance_factor: float

    def __init__(self,
                 space: StateSpace,
                 ranges: Tensor,
                 unit_noise: Callable[[Size], Tensor] = torch.randn,
                 x_0: Tensor = None) -> None:
        """Inits :py:class:`CenteredSampler` with specified distribution

        Args:
            space: Sampler's state space.
            ranges: ``(2 * space.n_v,)`` multiplicative scale on noise
              distribution standard deviation.
            unit_noise: Callback, returns coordinate-independent noise of
              nominal unit size.
            x_0: ``(space.n_x,)`` center of distribution, around which Lie
              algebra perturbation is applied by underlying
              :py:class:`WhiteNoiser`.
        """
        super().__init__(space)
        if x_0 is None:
            x_0 = space.zero_state()
        self.x_0 = space.project_state(x_0)

        self.noiser = WhiteNoiser(space, unit_noise)
        self.ranges = ranges

    def get_sample(self) -> Tensor:
        """Returns ``x_0`` distorted by white noise."""
        return self.noiser.noise(self.x_0, self.ranges)

    def covariance(self) -> Tensor:
        """Returns covariance of underlying noiser."""
        return self.noiser.covariance(self.ranges)


class UniformSampler(CenteredSampler):
    """Convenience :py:class:`CenteredSampler` for uniform noise."""

    def __init__(self, space: StateSpace, ranges: Tensor, x_0: Tensor = None):
        super().__init__(space, ranges, centered_uniform, x_0)


class GaussianSampler(CenteredSampler):
    """Convenience :py:class:`CenteredSampler` for Gaussian noise."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
