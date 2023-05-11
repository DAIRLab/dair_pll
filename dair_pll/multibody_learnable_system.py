"""Construction and analysis of learnable multibody systems.

Similar to Drake, multibody systems are instantiated as a child class of
:py:class:`System`: :py:class:`MultibodyLearnableSystem`. This object is a thin
wrapper for a :py:class:`MultibodyTerms` member variable, which manages
computation of lumped terms necessary for simulation and evaluation.

Simulation is implemented via Anitescu's [1] convex method.

An interface for the ContactNets [2] loss is also defined as an alternative
to prediction loss.

A large portion of the internal implementation of :py:class:`DrakeSystem` is
implemented in :py:class:`MultibodyPlantDiagram`.

[1] M. Anitescu, “Optimization-based simulation of nonsmooth rigid
multibody dynamics,” Mathematical Programming, 2006,
https://doi.org/10.1007/s10107-005-0590-7
[2] S. Pfrommer*, M. Halm*, and M. Posa. "ContactNets: Learning Discontinuous
Contact Dynamics with Smooth, Implicit Representations," Conference on
Robotic Learning, 2020, https://proceedings.mlr.press/v155/pfrommer21a.html
"""
from os import path
from typing import Tuple, Optional, Dict, cast

import numpy as np
import torch
from sappy import SAPSolver  # type: ignore
from torch import Tensor

from dair_pll import urdf_utils, tensor_utils, file_utils
from dair_pll.drake_system import DrakeSystem
from dair_pll.geometry import MeshRepresentation
from dair_pll.integrator import VelocityIntegrator
from dair_pll.multibody_terms import MultibodyTerms
from dair_pll.system import System, \
    SystemSummary
from dair_pll.tensor_utils import pbmm, broadcast_lorentz, project_lorentz, \
    reflect_lorentz


class MultibodyLearnableSystem(System):
    """:py:class:`System` interface for dynamics associated with
    :py:class:`MultibodyTerms`."""
    multibody_terms: MultibodyTerms
    init_urdfs: Dict[str, str]
    output_urdfs_dir: Optional[str] = None
    visualization_system: Optional[DrakeSystem]
    solver: SAPSolver
    dt: float

    def __init__(
        self,
        init_urdfs: Dict[str, str],
        dt: float,
        output_urdfs_dir: Optional[str] = None,
        learn_inertial_parameters: bool = False,
        mesh_representation: MeshRepresentation = MeshRepresentation
        .POLYGON
    ) -> None:
        """Inits :py:class:`MultibodyLearnableSystem` with provided model URDFs.

        Implementation is primarily based on Drake. Bodies are modeled via
        :py:class:`MultibodyTerms`, which uses Drake symbolics to generate
        dynamics terms, and the system can be exported back to a
        Drake-interpretable representation as a set of URDFs.

        Args:
            init_urdfs: Names and corresponding URDFs to model with
              :py:class:`MultibodyTerms`.
            dt: Time step of system in seconds.
            output_urdfs_dir: Optionally, a directory that learned URDFs can be
              written to.
            learn_inertial_parameters: Whether to learn inertial parameters.
            mesh_representation: Representation type for mesh geometry.
        """
        # pylint: disable=too-many-arguments
        multibody_terms = MultibodyTerms(init_urdfs,
                                         learn_inertial_parameters,
                                         mesh_representation)
        space = multibody_terms.plant_diagram.space
        integrator = VelocityIntegrator(space, self.sim_step, dt)
        super().__init__(space, integrator)
        self.multibody_terms = multibody_terms
        self.init_urdfs = init_urdfs
        self.output_urdfs_dir = output_urdfs_dir
        self.visualization_system = None
        self.solver = SAPSolver()
        self.dt = dt
        self.set_carry_sampler(lambda: Tensor([False]))
        self.max_batch_dim = 1

    def generate_updated_urdfs(self) -> Dict[str, str]:
        """Exports current parameterization as a :py:class:`DrakeSystem`.

        Returns:
            New Drake system instantiated on new URDFs.
        """
        assert self.output_urdfs_dir is not None
        old_urdfs = self.init_urdfs
        new_urdf_strings = urdf_utils.represent_multibody_terms_as_urdfs(
            self.multibody_terms, self.output_urdfs_dir)
        new_urdfs = {}

        # saves new urdfs with identical file basenames to original ones,
        # but in new folder.
        for urdf_name, new_urdf_string in new_urdf_strings.items():
            old_urdf_filename = path.basename(old_urdfs[urdf_name])
            new_urdf_path = path.join(self.output_urdfs_dir, old_urdf_filename)
            file_utils.save_string(new_urdf_path, new_urdf_string)
            new_urdfs[urdf_name] = new_urdf_path

        return new_urdfs

    def contactnets_loss_ncp(self,
                         x: Tensor,
                         u: Tensor,
                         x_plus: Tensor) -> Tensor:
        r"""Calculate ContactNets [1] loss for state transition.

        References:
            [1] S. Pfrommer*, M. Halm*, and M. Posa. "ContactNets: Learning
            Discontinuous Contact Dynamics with Smooth, Implicit
            Representations," Conference on Robotic Learning, 2020,
            https://proceedings.mlr.press/v155/pfrommer21a.html

        Args:
            x: (\*, space.n_x) current state batch.
            u: (\*, ?) input batch.
            x_plus: (\*, space.n_x) current state batch.

        Returns:
            (\*,) loss batch.
        """
        # pylint: disable-msg=too-many-locals
        v = self.space.v(x)
        q_plus, v_plus = self.space.q_v(x_plus)
        dt = self.dt
        eps = 1e-3

        delassus, M, J, phi, non_contact_acceleration = self.multibody_terms(
            q_plus, v_plus, u)

        n_contacts = phi.shape[-1]
        reorder_mat = tensor_utils.sappy_reorder_mat(n_contacts)
        reorder_mat = reorder_mat.reshape((1,) * (delassus.dim() - 2) +
                                          reorder_mat.shape).expand(
                                              delassus.shape)
        J_t = J[..., n_contacts:, :]

        # pylint: disable=E1103
        double_zero_vector = torch.zeros(phi.shape[:-1] + (2 * n_contacts,))
        phi_then_zero = torch.cat((phi, double_zero_vector), dim=-1)

        # pylint: disable=E1103
        sliding_velocities = pbmm(J_t, v_plus.unsqueeze(-1))
        sliding_speeds = sliding_velocities.reshape(phi.shape[:-1] +
                                                    (n_contacts, 2)).norm(
                                                        dim=-1, keepdim=True)

        Q = delassus + eps * torch.eye(3 * n_contacts)
        J_M = pbmm(reorder_mat.transpose(-1, -2),
                   pbmm(J, torch.linalg.cholesky(torch.inverse((M)))))

        dv = (v_plus - (v + non_contact_acceleration * dt)).unsqueeze(-2)

        q_pred = -pbmm(J, dv.transpose(-1, -2))
        q_comp = torch.abs(phi_then_zero).unsqueeze(-1)
        q_diss = torch.cat((sliding_speeds, sliding_velocities), dim=-2)
        q = q_pred + q_comp + q_diss

        penetration_penalty = (torch.maximum(
            -phi, torch.zeros_like(phi))**2).sum(dim=-1)

        penetration_penalty = penetration_penalty.reshape(
            penetration_penalty.shape + (1, 1))

        constant = 0.5 * pbmm(dv, pbmm(M, dv.transpose(
            -1, -2))) + penetration_penalty

        # Envelope theorem guarantees that gradient of loss w.r.t. parameters
        # can ignore the gradient of the force w.r.t. the QCQP parameters.
        # Therefore, we can detach ``force`` from pytorch's computation graph
        # without causing error in the overall loss gradient.
        # pylint: disable=E1103
        #force = self.solver.apply(Q, q, torch.rand(q.shape), 1e-7, 1000, 1e-7,
        #                          loss_pool).detach()
        force = pbmm(
            reorder_mat,
            self.solver.apply(
                J_M,
                pbmm(reorder_mat.transpose(-1, -2), q).squeeze(-1),
                eps).detach().unsqueeze(-1))

        # Hack: remove elements of ``force`` where solver likely failed.
        invalid = torch.any((force.abs() > 1e3) | force.isnan() | force.isinf(),
                            dim=-2,
                            keepdim=True)

        constant[invalid] *= 0.
        force[invalid.expand(force.shape)] = 0.

        loss = 0.5 * pbmm(force.transpose(-1, -2), pbmm(Q, force)) + pbmm(
            force.transpose(-1, -2), q) + constant

        return loss.squeeze(-1).squeeze(-1)

    def contactnets_loss_anitescu(self,
                         x: Tensor,
                         u: Tensor,
                         x_plus: Tensor) -> Tensor:
        r"""Variant of ContactNets [1] loss for Anitescu's formulation.

        References:
            [1] S. Pfrommer*, M. Halm*, and M. Posa. "ContactNets: Learning
            Discontinuous Contact Dynamics with Smooth, Implicit
            Representations," Conference on Robotic Learning, 2020,
            https://proceedings.mlr.press/v155/pfrommer21a.html

        Args:
            x: (\*, space.n_x) current state batch.
            u: (\*, ?) input batch.
            x_plus: (\*, space.n_x) current state batch.

        Returns:
            (\*,) loss batch.
        """
        # pylint: disable-msg=too-many-locals
        v = self.space.v(x)
        q_plus, v_plus = self.space.q_v(x_plus)
        dt = self.dt
        eps = 1e-3

        delassus, M, J, phi, non_contact_acceleration = self.multibody_terms(
            q_plus, v_plus, u)

        n_contacts = phi.shape[-1]
        reorder_mat = tensor_utils.sappy_reorder_mat(n_contacts)
        reorder_mat = reorder_mat.reshape((1,) * (delassus.dim() - 2) +
                                          reorder_mat.shape).expand(
                                              delassus.shape)
        J_t = J[..., n_contacts:, :]

        # pylint: disable=E1103
        #double_zero_vector = torch.zeros(phi.shape[:-1] + (2 * n_contacts,))
        #phi_then_zero = torch.cat((phi, double_zero_vector), dim=-1)
        sliding_velocities = pbmm(J_t, v_plus.unsqueeze(-1))
        phi_tilde = torch.cat(
            (phi, sliding_velocities * dt), dim=-1)
        reflected_phi_tilde = reflect_lorentz(phi_tilde)
        projected_reflected_phi_tilde = project_lorentz(reflected_phi_tilde)

        complementarity_penalty = project_lorentz(phi_tilde) + \
                                  projected_reflected_phi_tilde


        # pylint: disable=E1103

        sliding_speeds = sliding_velocities.reshape(phi.shape[:-1] +
                                                    (n_contacts, 2)).norm(
                                                        dim=-1, keepdim=True)

        Q = delassus + eps * torch.eye(3 * n_contacts)
        J_M = pbmm(reorder_mat.transpose(-1, -2),
                   pbmm(J, torch.linalg.cholesky(torch.inverse((M)))))

        dv = (v_plus - (v + non_contact_acceleration * dt)).unsqueeze(-2)

        q_pred = -pbmm(J, dv.transpose(-1, -2))
        q_comp = torch.abs(complementarity_penalty).unsqueeze(-1)
        q_diss = torch.cat((sliding_speeds, sliding_velocities), dim=-2)
        q = q_pred + q_comp + q_diss

        cone_penalty = (projected_reflected_phi_tilde**2).sum(dim=-1)

        cone_penalty = cone_penalty.reshape(
            cone_penalty.shape + (1, 1))

        constant = 0.5 * pbmm(dv, pbmm(M, dv.transpose(
            -1, -2))) + cone_penalty

        # Envelope theorem guarantees that gradient of loss w.r.t. parameters
        # can ignore the gradient of the force w.r.t. the QCQP parameters.
        # Therefore, we can detach ``force`` from pytorch's computation graph
        # without causing error in the overall loss gradient.
        # pylint: disable=E1103
        #force = self.solver.apply(Q, q, torch.rand(q.shape), 1e-7, 1000, 1e-7,
        #                          loss_pool).detach()
        force = pbmm(
            reorder_mat,
            self.solver.apply(
                J_M,
                pbmm(reorder_mat.transpose(-1, -2), q).squeeze(-1),
                eps).detach().unsqueeze(-1))

        # Hack: remove elements of ``force`` where solver likely failed.
        invalid = torch.any((force.abs() > 1e3) | force.isnan() | force.isinf(),
                            dim=-2,
                            keepdim=True)

        constant[invalid] *= 0.
        force[invalid.expand(force.shape)] = 0.

        loss = 0.5 * pbmm(force.transpose(-1, -2), pbmm(Q, force)) + pbmm(
            force.transpose(-1, -2), q) + constant

        return loss.squeeze(-1).squeeze(-1)

    def forward_dynamics(self,
                         q: Tensor,
                         v: Tensor,
                         u: Tensor) -> Tensor:
        r"""Calculates delta velocity from current state and input.

        Implement's Anitescu's [1] convex formulation in dual form, derived
        similarly to Tedrake [2] and described here.

        Let v_minus be the contact-free next velocity, i.e.::

            v + dt * non_contact_acceleration.

        Let FC be the combined friction cone::

            FC = {[beta_n beta_t]: beta_n_i >= ||beta_t_i||}.

        The primal version of Anitescu's formulation is as follows::

            min_{v_plus,s}  (v_plus - v_minus)^T M(q)(v_plus - v_minus)/2
            s.t.            s = [I; 0]phi(q)/dt + J(q)v_plus,
                            s \\in FC.

        The KKT conditions are the mixed cone complementarity
        problem [3, Theorem 2]::

            s = [I; 0]phi(q)/dt + J(q)v_plus,
            M(q)(v_plus - v_minus) = J(q)^T f,
            FC \\ni s \\perp f \\in FC.

        As M(q) is positive definite, we can solve for v_plus in terms of
        lambda, and thus these conditions can be simplified to::

            FC \\ni D(q)f + J(q)v_minus + [I;0]phi(q)/dt \\perp f \\in FC.

        which in turn are the KKT conditions for the dual QCQP we solve::

            min_{f}     f^T D(q) f/2 + f^T(J(q)v_minus + [I;0]phi(q)/dt)
            s.t.        f \\in FC.

        References:
            [1] M. Anitescu, “Optimization-based simulation of nonsmooth rigid
            multibody dynamics,” Mathematical Programming, 2006,
            https://doi.org/10.1007/s10107-005-0590-7
            [2] R. Tedrake. Underactuated Robotics: Algorithms for Walking,
            Running, Swimming, Flying, and Manipulation (Course Notes for MIT
            6.832), https://underactuated.mit.edu
            [3] S. Z. N'emeth, G. Zhang, "Conic optimization and
            complementarity problems," arXiv,
            https://doi.org/10.48550/arXiv.1607.05161
        Args:
            q: (\*, space.n_q) current configuration batch.
            v: (\*, space.n_v) current velocity batch.
            u: (\*, ?) current input batch.
            dynamics_pool: optional processing pool to enable multithreaded
              solves.

        Returns:
            (\*, space.n_v) delta velocity batch.
        """
        # pylint: disable=too-many-locals
        dt = self.dt
        eps = 1e6
        delassus, M, J, phi, non_contact_acceleration = self.multibody_terms(
            q, v, u)
        n_contacts = phi.shape[-1]
        contact_filter = (broadcast_lorentz(phi) <= eps).unsqueeze(-1)
        contact_matrix_filter = pbmm(contact_filter.int(),
                                     contact_filter.transpose(-1,
                                                              -2).int()).bool()

        reorder_mat = tensor_utils.sappy_reorder_mat(n_contacts)
        reorder_mat = reorder_mat.reshape((1,) * (delassus.dim() - 2) +
                                          reorder_mat.shape).expand(
                                              delassus.shape)
        J_M = pbmm(reorder_mat.transpose(-1, -2),
                   pbmm(J, torch.linalg.cholesky(torch.inverse((M)))))

        # pylint: disable=E1103
        double_zero_vector = torch.zeros(phi.shape[:-1] + (2 * n_contacts,))
        phi_then_zero = torch.cat((phi, double_zero_vector),
                                  dim=-1).unsqueeze(-1)
        # pylint: disable=E1103
        Q_full = delassus + torch.eye(3 * n_contacts) * 1e-4

        v_minus = v + dt * non_contact_acceleration
        q_full = pbmm(J, v_minus.unsqueeze(-1)) + (1 / dt) * phi_then_zero

        Q = torch.zeros_like(Q_full)
        q = torch.zeros_like(q_full)
        Q[contact_matrix_filter] += Q_full[contact_matrix_filter]
        q[contact_filter] += q_full[contact_filter]

        impulse_full = pbmm(
            reorder_mat,
            self.solver.apply(
                J_M,
                pbmm(reorder_mat.transpose(-1, -2), q_full).squeeze(-1),
                1e-4).unsqueeze(-1))

        impulse = torch.zeros_like(impulse_full)
        impulse[contact_filter] += impulse_full[contact_filter]

        return v_minus + torch.linalg.solve(M, pbmm(J.transpose(-1, -2),
                                                    impulse)).squeeze(-1)

    def sim_step(self, x: Tensor, carry: Tensor) -> Tuple[Tensor, Tensor]:
        """``Integrator.partial_step`` wrapper for
        :py:meth:`forward_dynamics`."""
        q, v = self.space.q_v(x)
        # pylint: disable=E1103
        u = torch.zeros(q.shape[:-1] + (0,))
        v_plus = self.forward_dynamics(q, v, u)
        return v_plus, carry

    def summary(self, statistics: Dict) -> SystemSummary:
        """Generates summary statistics for multibody system.

        The scalars returned are simply the scalar description of the
        system's :py:class:`MultibodyTerms`.

        Meshes are generated for learned
        :py:class:`~dair_pll.geometry.DeepSupportConvex` es.

        Args:
            statistics: Updated evaluation statistics for the model.

        Returns:
            Scalars and meshes packaged into a ``SystemSummary``.
        """
        scalars, meshes = self.multibody_terms.scalars_and_meshes()
        videos = cast(Dict[str, Tuple[np.ndarray, int]], {})

        return SystemSummary(scalars=scalars, videos=videos, meshes=meshes)
