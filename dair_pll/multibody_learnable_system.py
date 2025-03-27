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

# pylint: disable=invalid-name,too-many-statements,too-many-locals,too-many-lines
# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-positional-arguments

from contextlib import nullcontext
from dataclasses import dataclass
from os import path
import pdb
from itertools import chain
from typing import Any, Callable, List, Iterable, Tuple, Optional, Dict, cast, Union

import gin
import numpy as np
import torch
from torch import Tensor
from torch.autograd.functional import hessian
from torch.nn import Parameter
from torch.distributions.normal import Normal
#from torch.distributions.gamma import Gamma
from tensordict.tensordict import TensorDictBase, TensorDict
from torch.utils.data import DataLoader

from dair_pll import urdf_utils, tensor_utils, file_utils
from dair_pll.drake_system import DrakeSystem
from dair_pll.integrator import VelocityIntegrator
from dair_pll.learnable_trajectory import LearnableTrajectories
from dair_pll.multibody_terms import MultibodyTerms, LearnableBodySettings
from dair_pll.solvers import DynamicCvxpyLCQPLayer
from dair_pll.state_space import StateSpace, ProductSpace
from dair_pll.system import SystemSummary
from dair_pll.tensor_utils import pbmm, broadcast_lorentz

from dair_pll.drake_utils import (
    unique_body_identifier,
    get_bodies_in_model_instance,
)

from dair_pll.geometry import CollisionGeometry, PydrakeToCollisionGeometryFactory
from pydrake.all import Shape

@gin.configurable("LearnableHyperparameters")
@dataclass
class MultibodyLearnableSystemHyperparameters:
    """Class to specify hyperparameters"""
    w_pred: float = 1e0
    w_q_pred: float = 1e0
    w_comp: float = 1e0
    w_fdiss: float = 1e0
    w_ndiss: float = 1e0
    w_pen: float = 1e0
    w_dev: float = 1e0
    w_norm: float = 1e0
    w_reg_iner: float = 1e0
    n_fisher_samples: int = 10

@gin.configurable
class MultibodyLearnableSystem(DrakeSystem):
    """:py:class:`System` interface for dynamics associated with
    :py:class:`MultibodyTerms`."""

    multibody_terms: MultibodyTerms
    init_urdfs: Dict[str, str]
    output_urdfs_dir: Optional[str] = None
    _solver: DynamicCvxpyLCQPLayer
    _hyperparameters: MultibodyLearnableSystemHyperparameters
    _default_dt: float
    loss_cache: Dict[str, Any]

    def __init__(
        self,
        init_urdfs: Dict[str, str],
        hyperparameters: MultibodyLearnableSystemHyperparameters = MultibodyLearnableSystemHyperparameters(),
        learnable_body_dict: Optional[Dict[str, LearnableBodySettings]] = None,
        default_dt: float = 0.0333,
        output_urdfs_dir: Optional[str] = None,
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
            learnable_body_dict: dict of body names and which properties should
              be learned
            output_urdfs_dir: Optionally, a directory that learned URDFs can be
              written to.
        """
        if learnable_body_dict is None:
            learnable_body_dict = {}

        multibody_terms = MultibodyTerms(
            init_urdfs,
            learnable_body_dict,
        )

        # Init Parent System
        space = multibody_terms.plant_diagram.space
        integrator = VelocityIntegrator(space, self.sim_step, default_dt)
        super(DrakeSystem, self).__init__(space, integrator)

        self.output_urdfs_dir = output_urdfs_dir
        self.multibody_terms = multibody_terms
        self.init_urdfs = init_urdfs
        self.urdfs = init_urdfs

        # TODO: HACK re-add random initialization

        # Pylint doesn't know about gin
        # pylint: disable=no-value-for-parameter
        self._solver = DynamicCvxpyLCQPLayer()
        self._default_dt = default_dt
        self.set_carry_sampler(lambda: torch.tensor([False]))
        self.max_batch_dim = 1
        self._hyperparameters = hyperparameters

        # Match DrakeSystem Attributes
        self.plant_diagram = multibody_terms.plant_diagram

        self.loss_cache = {}
        self.debug = False

    def generate_updated_urdfs(self, suffix: str = None) -> Dict[str, str]:
        """Exports current parameterization as a :py:class:`DrakeSystem`.

        Returns:
            New Drake system instantiated on new URDFs.
        """
        assert self.output_urdfs_dir is not None
        new_urdf_strings = urdf_utils.represent_multibody_terms_as_urdfs(
            self.multibody_terms, self.output_urdfs_dir
        )

        # saves new urdfs with model name plus optional suffix in
        # new folder.
        for urdf_name, new_urdf_string in new_urdf_strings.items():
            new_urdf_filename = urdf_name + ".urdf"
            if suffix is not None:
                new_urdf_filename = (
                    new_urdf_filename.split(".")[0] + "_" + suffix + ".urdf"
                )

            new_urdf_path = path.join(self.output_urdfs_dir, new_urdf_filename)
            file_utils.save_string(new_urdf_path, new_urdf_string)

        self.urdfs = new_urdf_strings
        return new_urdf_strings

    def contactnets_loss(
        self,
        x: Tensor,
        u: Tensor,
        x_plus: Tensor,
        contact_forces: Optional[Dict[Tuple[str, str], Tensor]] = None,
        contact_normals: Optional[Dict[Tuple[str, str], Tensor]] = None,
        impulses: Optional[Tensor] = None,
        use_envelope: bool = True,
    ) -> Tensor:
        r"""Calculate ContactNets [1] loss for state transition.

        Change made to scale this loss to be per kilogram.  This helps prevent
        sending mass quantities to zero in multibody learning scenarios.

        References:
            [1] S. Pfrommer*, M. Halm*, and M. Posa. "ContactNets: Learning
            Discontinuous Contact Dynamics with Smooth, Implicit
            Representations," Conference on Robotic Learning, 2020,
            https://proceedings.mlr.press/v155/pfrommer21a.html

        Args:
            x: (\*, space.n_x) current state batch.
            u: (\*, ?) input batch.
            x_plus: (\*, space.n_x) current state batch.
            contact_forces: mapping (obj_a_name, obj_b_name) to force on obj_b in World Frame
            contact_normals: mapping (obj_a_name, obj_b_name) to surface normal towards obj_b in World Frame

        Returns:
            (\*,) loss batch.
        """
        loss_pred, loss_q_pred, loss_comp, loss_pen, loss_fdiss, loss_ndiss, loss_dev, loss_norm = (
            self.calculate_contactnets_loss_terms(
                x, u, x_plus, contact_forces, contact_normals, impulses, use_envelope=use_envelope
            )
        )

        # regularizers = self.get_regularization_terms(x, u, x_plus)

        # For now the regularization terms are: 0) inertia matrix condition number.
        # Will need to be updated later if more are added.
        # reg_inertia_cond = regularizers[0]

        loss = (
            (self._hyperparameters.w_pred * loss_pred)
            + (self._hyperparameters.w_q_pred * loss_q_pred)
            + (self._hyperparameters.w_comp * loss_comp)
            + (self._hyperparameters.w_pen * loss_pen)
            + (self._hyperparameters.w_fdiss * loss_fdiss)
            + (self._hyperparameters.w_ndiss * loss_ndiss)
            + (self._hyperparameters.w_dev * loss_dev)
            + (self._hyperparameters.w_norm * loss_norm)
            # TODO: HACK re-add later
            #            + (self._hyperparameters.w_reg_iner * reg_inertia_cond)
        )

        # Cache Losses
        self.loss_cache["loss_pred"] = loss_pred.clone().detach()
        self.loss_cache["loss_q_pred"] = loss_q_pred.clone().detach()
        self.loss_cache["loss_comp"] = loss_comp.clone().detach()
        self.loss_cache["loss_pen"] = loss_pen.clone().detach()
        self.loss_cache["loss_fdiss"] = loss_fdiss.clone().detach()
        self.loss_cache["loss_ndiss"] = loss_ndiss.clone().detach()
        self.loss_cache["loss_dev"] = loss_dev.clone().detach()
        self.loss_cache["loss_norm"] = loss_norm.clone().detach()

        return loss

    def get_regularization_terms(
        self, x: Tensor, u: Tensor, x_plus: Tensor, **kwargs
    ) -> List[Tensor]:
        """Calculate some regularization terms."""

        regularizers = []

        # Penalize the condition number of the mass matrix.
        q_plus, v_plus = self.space.q_v(x_plus)
        _, M, _, _, _, _, _, _ = self.get_multibody_terms(q_plus, v_plus, u)
        # TODO HACK: hard-coded. rows/cols in M should match  model_body_qw in GetStateNames()
        I_BBcm_B = M[..., 3:6, 3:6]
        # pylint doesn't know about torch functions
        # pylint: disable=E1102
        regularizers.append(torch.linalg.cond(I_BBcm_B))

        # TODO: Use the believed geometry to help supervise the learned CoM.
        return regularizers

    def calculate_contactnets_loss_terms(
        self,
        x: Tensor,
        u: Tensor,
        x_plus: Tensor,
        contact_forces: Optional[Dict[Tuple[str, str], Tensor]] = None,
        contact_normals: Optional[Dict[Tuple[str, str], Tensor]] = None,
        impulses: Optional[Tensor] = None,
        ret_impulse_only: bool = False,
        use_envelope: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Helper function for
        :py:meth:`MultibodyLearnableSystem.contactnets_loss` that returns the
        individual pre-weighted loss contributions:

            * Prediction
            * Complementarity
            * Penetration
            * Dissipation

        Args:
            x: (*, space.n_x) current state batch.
            u: (*, ?) input batch.
            x_plus: (*, space.n_x) current state batch.

        Returns:
            (*,) prediction error loss.
            (*,) prediction (q_plus = q + v) error loss
            (*,) complementarity violation loss.
            (*,) penetration loss.
            (*,) dissipation violation loss.
            (*,) deviation from measurement loss
        """
        batch_dims = x.size()[:-1]
        assert x.size() == x_plus.size()
        v = self.space.v(x)
        q_plus, v_plus = self.space.q_v(x_plus)
        dt = self._default_dt
        eps = 1e-8  # TODO: HACK, make a hyperparameter
        
        # Begin loss calculation.
        (
            _, # Delassus
            M,
            J,
            phi,
            non_contact_acceleration,
            obj_pair_list,
            R_FW_list,
            mu_list,
        ) = self.get_multibody_terms(q_plus, v_plus, u, contact_normals)

        # Prepare to exclude the robot predictions from the prediction loss.
        # First n_q of self.state_map_for_learnable_bodies: states; the rest (last n_v): velocities.
        # velocity_mask: 1 for object velocities, 0 for robot velocities.
        # object velocities: wx, wy, wz, vx, vy, vz
        velocity_mask = self.state_map_for_learnable_bodies()[self.space.n_q:].detach().clone()
        J_small = J[..., velocity_mask] # (*, n_contacts*3, n_v_object)
        M_small = M[..., velocity_mask, :][..., velocity_mask]
        M_inv_small = torch.inverse(M_small)
        delassus = pbmm(J_small, pbmm(M_inv_small, J_small.transpose(-1, -2)))

        if contact_forces is None:
            contact_forces = {}
        if contact_normals is None:
            contact_normals = {}

        # Construct a reordering matrix s.t. lambda_CN = reorder_mat @ f_sappy.
        n_contacts = phi.shape[-1]
        reorder_mat = tensor_utils.sappy_reorder_mat(n_contacts)
        reorder_mat = reorder_mat.reshape(
            (1,) * (delassus.dim() - 2) + reorder_mat.shape
        ).expand(delassus.shape)

        double_zero_vector = torch.zeros(phi.shape[:-1] + (2 * n_contacts,))
        phi_then_zero = torch.cat((phi, double_zero_vector), dim=-1)

        J_t = J[..., n_contacts:, :]
        sliding_velocities = pbmm(J_t, v_plus.unsqueeze(-1))

        ### Need non-0 norm for Hessian calculation
        sliding_vels_reshape = sliding_velocities.reshape(
            phi.shape[:-1] + (n_contacts, 2)
        )
        sliding_eps = torch.ones_like(sliding_vels_reshape) * eps
        sliding_speeds = (sliding_vels_reshape + sliding_eps).norm(dim=-1, keepdim=True)

        J_n = J[..., :n_contacts, :]
        normal_velocities = pbmm(J_n, v_plus.unsqueeze(-1))
        normal_velocities = torch.maximum(
            normal_velocities, torch.zeros_like(normal_velocities)
        )

        # Units: Energy
        Q_delassus = delassus + eps * torch.eye(3 * n_contacts)  # Force PD

        dv = (v_plus - (v + non_contact_acceleration * dt)).unsqueeze(-2)
        dv_small = dv[..., velocity_mask]

        # Constant Terms
        # Calculate the prediction constant based on loss formulation mode.
        constant_pred = 0.5 * pbmm(dv_small, pbmm(M_small, dv_small.transpose(-1, -2)))
        constant_pen = torch.square(torch.maximum(-phi, torch.zeros_like(phi))).sum(dim=-1)
        constant_pen = constant_pen.reshape(constant_pen.shape + (1, 1))

        # Calculate q vectors
        # Final Units: Energy -> q units velocity
        q_pred = -pbmm(J_small, dv_small.transpose(-1, -2))
        q_comp = (1.0 / dt) * torch.square(torch.maximum(
            phi_then_zero, torch.zeros_like(phi_then_zero))
        ).unsqueeze(-1)
        q_diss = torch.cat((sliding_speeds, sliding_velocities), dim=-2)
        q_n_diss = torch.cat(
            (normal_velocities, double_zero_vector.unsqueeze(-1)), dim=-2
        )

        # Penalize Deviation from measured contact impulses
        # This is in impulse^2. TODO: take deviation w.r.t. Delassus to
        # add 1/mass term to bring into Energy.
        q_dev = torch.zeros_like(q_pred)
        Q_dev = torch.zeros_like(Q_delassus)
        constant_dev = torch.zeros_like(constant_pred)

        # Penalize Normal Deviation
        # This is unitless. TODO: figure out energy conversion.
        q_norm = torch.zeros_like(q_pred)
        for key in contact_normals.keys():
            indices = np.array([i for i, x in enumerate(obj_pair_list) if x == key])
            if len(indices) == 0:
                continue
            for idx in indices:
                R_FW = R_FW_list[idx]
                normals_guess_W = R_FW.transpose(-1, -2)[..., 2]
                assert normals_guess_W.size() == batch_dims + (3,)
                assert contact_normals[key].size() == batch_dims + (3,)
                nonzero_norm = (torch.linalg.vector_norm(contact_normals[key], dim=-1) > 0.)
                # if contact_normals is 0, then cost = 0 i.e. align with guess
                normals_measured_W = normals_guess_W.clone().detach()
                normals_measured_W[nonzero_norm] = torch.nn.functional.normalize(contact_normals[key][nonzero_norm], dim=-1)
                assert normals_measured_W.size() == batch_dims + (3,)
                # Batch dot product (max 0 for numerical stability)
                cost_normal = torch.maximum(1.0 - (normals_measured_W * normals_guess_W).sum(dim=-1), torch.zeros(batch_dims))
                assert cost_normal.size() == batch_dims
                q_norm[..., idx, 0] = cost_normal

        for key in contact_forces.keys():
            indices = np.array([i for i, x in enumerate(obj_pair_list) if x == key])
            if len(indices) == 0:
                continue
            mu_i = mu_list[indices[0]]
            # Q_dev = diag(mu)RS^TSR^Tdiag(mu)^T; diag(mu) = 1 if normal, mu otherwise
            # R is block diagonal rotation matrices, S is summation matrix
            diag_F_mu = torch.zeros(
                q_dev.shape[:-1] + ((len(indices) * 3),)
            )  # (batch x (n_c_tot*3) x (n_c_obj*3))
            R_FW_mat = torch.zeros(
                q_dev.shape[:-2] + ((len(indices) * 3), (len(indices) * 3))
            )  # (batch x (n_c_obj*3) x (n_c_obj*3))
            sum_W_mat = torch.zeros(
                q_dev.shape[:-2] + (3, (len(indices) * 3))
            )  # (batch x 3 x (n_c_obj*3))
            for contact, idx in enumerate(indices):

                # Map Normal Force
                diag_F_mu[..., idx, contact * 3 + 2] = 1.0
                # Map Tangent Forces
                diag_F_mu[..., len(obj_pair_list) + 2 * idx, contact * 3] = mu_i
                diag_F_mu[..., len(obj_pair_list) + 2 * idx + 1, contact * 3 + 1] = mu_i

                # Create Block diagonal matrix (note torch.block_diag isn't vectorized)
                R_FW_mat[
                    ...,
                    contact * 3 : (contact + 1) * 3,
                    contact * 3 : (contact + 1) * 3,
                ] = R_FW_list[idx]

                # Summation
                sum_W_mat[..., 0, contact * 3] = 1.0
                sum_W_mat[..., 1, contact * 3 + 1] = 1.0
                sum_W_mat[..., 2, contact * 3 + 2] = 1.0
            q_dev_part = pbmm(
                sum_W_mat, pbmm(R_FW_mat.transpose(-1, -2), diag_F_mu.transpose(-1, -2))
            )  # (batch, 3, (n_c_tot*3))
            Q_dev += pbmm(q_dev_part.transpose(-1, -2), q_dev_part)

            # Linear Term is lambda_mSR^Tdiag(mu)^T
            impulse_measured_W = contact_forces[key].unsqueeze(-2) * dt  # (batch, 1, 3)
            q_dev -= pbmm(impulse_measured_W, q_dev_part).transpose(
                -1, -2
            )  # (batch, n_c_tot*3, 1)

            # Constant term is lambda_m magnitude, multiply by 0.5 here to match constant_pred
            constant_dev += 0.5 * pbmm(
                impulse_measured_W, impulse_measured_W.transpose(-1, -2)
            )

        Q_final = Q_delassus + (self._hyperparameters.w_dev / self._hyperparameters.w_pred) * Q_dev

        q_final = (
            q_pred
            + (self._hyperparameters.w_comp / self._hyperparameters.w_pred) * q_comp
            + (self._hyperparameters.w_norm / self._hyperparameters.w_pred) * q_norm
            + (self._hyperparameters.w_fdiss / self._hyperparameters.w_pred) * q_diss
            + (self._hyperparameters.w_ndiss / self._hyperparameters.w_pred) * q_n_diss
            + (self._hyperparameters.w_dev / self._hyperparameters.w_pred) * q_dev
        )

        # Envelope theorem guarantees that gradient of loss w.r.t. parameters
        # can ignore the gradient of the impulses w.r.t. the QCQP parameters.
        # Therefore, we can detach ``impulses`` from pytorch's computation graph
        # without causing error in the overall loss gradient.
        if impulses is None:
            with torch.no_grad() if use_envelope else nullcontext():
                impulses = pbmm(
                    reorder_mat,
                    self._solver(
                        pbmm(
                            reorder_mat.transpose(-1, -2), pbmm(Q_final, reorder_mat)
                        ),  # Quadratic Term
                        pbmm(reorder_mat.transpose(-1, -2), q_final).squeeze(
                            -1
                        ),  # Linear Term
                    ).unsqueeze(-1),
                )

        # Hack: remove elements of ``impulses`` where solver likely failed.
        invalid = torch.any(
            (impulses.abs() > 1e3) | impulses.isnan() | impulses.isinf(),
            dim=-2,
            keepdim=True,
        )

        constant_pen[invalid] *= 0.0
        constant_pred[invalid] *= 0.0
        constant_dev[invalid] *= 0.0
        impulses[invalid.expand(impulses.shape)] = 0.0
        # Zero out negative normals (possible within solver tolerance)
        impulses[..., :n_contacts, 0] = torch.maximum(impulses[..., :n_contacts, 0].clone(), torch.zeros_like(impulses[..., :n_contacts, 0].detach())).clone()

        if ret_impulse_only:
            return impulses.squeeze(-1)

        loss_pred = (
            0.5 * pbmm(impulses.transpose(-1, -2), pbmm(Q_delassus, impulses))
            + pbmm(impulses.transpose(-1, -2), q_pred)
            + constant_pred
        )
        vel_err = (
            self.space.configuration_difference(self.space.euler_step(self.space.q(x), self.space.v(x), dt), self.space.q(x_plus))
            / dt
        ).unsqueeze(-1)
        loss_q_pred = pbmm(vel_err.transpose(-1, -2), pbmm(M, vel_err))
        loss_comp = pbmm(impulses.transpose(-1, -2), q_comp)
        loss_pen = constant_pen
        loss_fdiss = pbmm(impulses.transpose(-1, -2), q_diss)
        loss_ndiss = pbmm(
            impulses.transpose(-1, -2), q_n_diss
        )
        loss_dev = (
            0.5 * pbmm(impulses.transpose(-1, -2), pbmm(Q_dev, impulses))
            + pbmm(impulses.transpose(-1, -2), q_dev)
            + constant_dev
        )
        loss_norm = pbmm(impulses.transpose(-1, -2), q_norm)

        # Interpretable Loss Terms
        self.loss_cache["mean_dev_N"] = (
            torch.sqrt(loss_dev.clone().detach().mean()) / dt
        )
        self.loss_cache["mean_diss_Jps"] = loss_fdiss.clone().detach().mean() / dt
        self.loss_cache["mean_comp_Nm"] = loss_comp.clone().detach().mean()
        self.loss_cache["mean_pen_m"] = torch.sqrt(loss_pen.clone().detach().mean())
        self.loss_cache["mean_q_pred_mps"] = torch.sqrt(
            pbmm(vel_err.transpose(-1, -2), vel_err).clone().detach().mean()
        )
        self.loss_cache["mean_pred_Nm"] = loss_pred.clone().detach().mean()
        self.loss_cache["mean_norm_cosine"] = q_norm.clone().detach().mean()

        # Check for positive definite loss
        try:
            assert np.all(loss_dev.detach().cpu().numpy() >= 0.0), "Deviation Loss Negative"
            assert np.all(loss_pred.detach().cpu().numpy() >= 0.0), "Prediction Loss Negative"
            assert np.all(loss_norm.detach().cpu().numpy() >= 0.0), "Normal Alignment Loss Negative"
        except AssertionError:
            # pylint: disable-next=forgotten-debug-statement
            pdb.Pdb(nosigint=True).set_trace()

        if self.debug:
            breakpoint()

        return (
            loss_pred.reshape(batch_dims),
            loss_q_pred.reshape(batch_dims),
            loss_comp.reshape(batch_dims),
            loss_pen.reshape(batch_dims),
            loss_fdiss.reshape(batch_dims),
            loss_ndiss.reshape(batch_dims),
            loss_dev.reshape(batch_dims),
            loss_norm.reshape(batch_dims),
        )

    def get_multibody_terms(
        self,
        q: Tensor,
        v: Tensor,
        u: Tensor,
        estimated_normals_W: Optional[Dict[Tuple[str, str], Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, List, List]:
        """Get multibody terms of the system.  Without a residual, this is a
        straightfoward pass-through to the system's :py:class:`MultibodyTerms`.
        With a residual, the residual augments the continuous dynamics."""

        if estimated_normals_W is None:
            estimated_normals_W = {}

        (
            delassus,
            M,
            J,
            phi,
            non_contact_acceleration,
            obj_pair_list,
            R_FW_list,
            mu_list,
        ) = self.multibody_terms(q, v, u, estimated_normals_W)

        return (
            delassus,
            M,
            J,
            phi,
            non_contact_acceleration,
            obj_pair_list,
            R_FW_list,
            mu_list,
        )

    def forward_dynamics(
        self,
        q: Tensor,
        v: Tensor,
        u: Tensor,
        dts: Optional[Union[float, Tensor]] = None,
    ) -> Tensor:
        r"""Calculates delta velocity from current state and input.

        Implements Anitescu's [1] convex formulation in dual form, derived
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

        Returns:
            (\*, space.n_v) delta velocity batch.
        """
        # pylint: disable=too-many-locals
        dt = self._default_dt if dts is None else dts
        phi_eps = 1e-3
        eps = 1e-8  # TODO: HACK make this a hyperparameter
        delassus, M, J, phi, non_contact_acceleration, obj_pair_list, R_FW_list, mu_list = (
            self.get_multibody_terms(q, v, u)
        )
        n_contacts = phi.shape[-1]
        contact_filter = (broadcast_lorentz(phi) <= phi_eps).unsqueeze(-1)

        reorder_mat = tensor_utils.sappy_reorder_mat(n_contacts)
        reorder_mat = reorder_mat.reshape(
            (1,) * (delassus.dim() - 2) + reorder_mat.shape
        ).expand(delassus.shape)

        Q_delassus = delassus + eps * torch.eye(3 * n_contacts)

        # pylint: disable=E1103
        double_zero_vector = torch.zeros(phi.shape[:-1] + (2 * n_contacts,))
        phi_then_zero = torch.cat((phi, double_zero_vector), dim=-1).unsqueeze(-1)

        v_minus = v + dt * non_contact_acceleration
        q_full = pbmm(J, v_minus.unsqueeze(-1)) + (1 / dt) * phi_then_zero

        with torch.no_grad():
            impulse_full = pbmm(
                reorder_mat,
                self._solver(
                    pbmm(
                        reorder_mat.transpose(-1, -2), pbmm(Q_delassus, reorder_mat)
                    ),  # Quadratic Term
                    pbmm(reorder_mat.transpose(-1, -2), q_full).squeeze(
                        -1
                    ),  # Linear Term
                ).unsqueeze(-1),
            )

            impulse = torch.zeros_like(impulse_full)
            # Clamp to avoid small negative normal force
            impulse_full[..., :n_contacts, :] = impulse_full[..., :n_contacts, :].clamp(min=0.)
            impulse[contact_filter] += impulse_full[contact_filter].detach()

            v_add = torch.linalg.solve(M, pbmm(J.transpose(-1, -2), impulse)).squeeze(-1).detach()

        ### Construct contact forces / normals
        batch_dims = q.size()[:-1]
        ret_contact_forces = {} # Dict[Tuple[str, str], Tensor]
        ret_contact_normals = {} # Dict[Tuple[str, str], Tensor]
        for key in obj_pair_list:
            if obj_pair_list.count(key) == 1:
                ret_contact_forces[key] = torch.zeros(batch_dims + (1, 3))
                ret_contact_normals[key] = torch.zeros(batch_dims + (1, 3))

        for key in ret_contact_forces.keys():
            indices = np.array([i for i, x in enumerate(obj_pair_list) if x == key])
            if len(indices) == 0:
                continue
            assert len(indices) == 1
            index = indices[0]
            fric_index = len(obj_pair_list) + 2 * index
            R_WF_i = R_FW_list[index].transpose(-1, -2).detach()
            ret_contact_normals[key][contact_filter[..., index, 0], 0, :] = R_WF_i[contact_filter[..., index, 0], :, 2]
            # Force in contact_frame
            ret_contact_forces[key][..., 0, 0] = impulse[..., index, 0] / dt
            ret_contact_forces[key][..., 0, 1:] = mu_list[index].detach() * impulse[..., fric_index:fric_index+2, 0] / dt
            # Rotate into world frame
            ret_contact_forces[key] = pbmm(R_WF_i, ret_contact_forces[key].transpose(-1, -2)).transpose(-1, -2)      
        ###

        # pylint doesn't know about torch functions
        # pylint: disable=E1102
        ## TODO: HACK Only differentiate euler steps.
        return (
            v_minus + v_add,
            ret_contact_forces,
            ret_contact_normals,
        )

    def sim_step(self, x: Tensor, carry: Tensor) -> Tuple[Tensor, Tensor]:
        """``Integrator.partial_step`` wrapper for
        :py:meth:`forward_dynamics`."""
        q, v = self.space.q_v(x)
        # pylint: disable=E1103
        u = torch.zeros(q.shape[:-1] + (0,))
        v_plus, _ = self.forward_dynamics(q, v, u)
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


@gin.configurable("LearnableSystem")
class MultibodyLearnableSystemWithTrajectory(MultibodyLearnableSystem):
    """:py:class:`MultibodyLearnableSystem` where a model can have
    learnable trajectories."""

    _model_spaces: Dict[str, StateSpace]
    r"""Map of model name to state space, ignoring spaces where n_x == 0"""
    _trajectory_model_names: List[str]
    r"""Name of the model corresponding to the trajectory"""
    _trajectory: LearnableTrajectories
    r"""The learnable trajectory for the given models"""

    def __init__(
        self,
        trajectory_model_names: Union[List[str], str],
        init_traj_state: Optional[Union[List[float], Tensor]] = None,
        **kwargs,
    ) -> None:
        ## Construct Super System
        super().__init__(**kwargs)
        self._trajectory_model_names = (
            trajectory_model_names
            if isinstance(trajectory_model_names, list)
            else [trajectory_model_names]
        )

        ## Populate Model Spaces
        self._model_spaces = {}
        traj_spaces = []
        plant_diagram = self.multibody_terms.plant_diagram
        for model_id, space in zip(plant_diagram.model_ids, plant_diagram.space.spaces):
            name = plant_diagram.plant.GetModelInstanceName(model_id)
            self._model_spaces[name] = space
            if name in self._trajectory_model_names:
                traj_spaces.append(space)

        ## Create Trajectory Parameters
        init_state = None
        if init_traj_state is not None:
            init_state = (
                init_traj_state
                if isinstance(init_traj_state, Tensor)
                else torch.tensor(init_traj_state)
            )
        self._trajectory = LearnableTrajectories(ProductSpace(traj_spaces), init_state)

    def get_model_space(self, model_name: str) -> Optional[StateSpace]:
        """Getter for model space"""
        if model_name in self._model_spaces:
            return self._model_spaces[model_name]
        return None

    @gin.register
    def state_map_for_learnable_bodies(
        self,
        robot_model_name: str = "robot"
    ) -> Tensor:
        """Returns a boolean tensor indicating which states correspond to
        learnable bodies.
        """
        # state_names: robot joint states, object states, 
        # robot joint velocities, object velocities
        state_names = self.multibody_terms.plant_diagram.plant.GetStateNames()
        return torch.tensor([not s.startswith(robot_model_name) for s in state_names])

    @gin.configurable
    def diff_simulate(
        self,
        robot_target_trajectories: Tensor,
        timestamps: Tensor,
        robot_model_name: str = "robot",
        kp: float = 20.0,
        kd: float = 10.0,
        steps_per_timestep=1,
    ) -> Tensor:
        """
        From the current estimated model state, simulate a batch of robots on the target trajectories.

        Args:
            robot_target_trajectories: (batch, traj_len, robot_nx)
            timestamps: (traj_len,)
        Returns:
            - model_states_from_state_tensor() of (batch, traj_len, plant.n_x)
            - impulse_star  (batch, traj_len, n_collisions)
            - robot u  (batch, traj_len, robot.n_v)
        """

        # Input Validation
        robot_space = self._model_spaces[robot_model_name]
        assert robot_space.n_q == robot_space.n_v
        assert len(robot_target_trajectories.size()) >= 3
        batch_dims = robot_target_trajectories.size()[:-2]
        traj_len = robot_target_trajectories.size()[-2]
        assert robot_target_trajectories.size() == batch_dims + (
            traj_len,
            robot_space.n_x,
        ), str(robot_target_trajectories.size())
        assert timestamps.size() == (traj_len,)

        # Populate Initial State
        data_state = TensorDict({}, batch_size=batch_dims + (traj_len,))
        data_state[robot_model_name + "_state"] = torch.zeros_like(
            robot_target_trajectories
        )
        data_state[robot_model_name + "_state"][..., 0, :] = robot_target_trajectories[
            ..., 0, :
        ]
        # Zero out current state velocity
        current_x = self._trajectory.current_state()
        traj_splits = self._trajectory.space.x_split(current_x)
        for traj_model_idx, traj_model_name in enumerate(self._trajectory_model_names):
            model_x = traj_splits[traj_model_idx]
            assert len(model_x.size()) == 1
            data_state[traj_model_name + "_state"] = torch.zeros(
                batch_dims + (traj_len, model_x.size()[0])
            )
            data_state[traj_model_name + "_state"][..., 0, :] = model_x

        plant_states = super().construct_state_tensor(data_state)
        assert plant_states.size() == batch_dims + (traj_len, self.space.n_x)

        ## Simulation Loop
        ret_contact_forces = {}
        ret_contact_normals = {}
        ret_u = torch.zeros(batch_dims + (traj_len, robot_space.n_v))
        for sim_idx in range(1, traj_len):
            print(f"Sim Step {sim_idx} / {traj_len}...")
            sim_dt = timestamps[sim_idx] - timestamps[sim_idx - 1]
            step_dt = sim_dt / float(steps_per_timestep)
            step_states = torch.zeros(batch_dims + (steps_per_timestep+1, self.space.n_x))
            step_states[..., 0, :] = plant_states[..., sim_idx - 1, :]
            for step_idx in range(1, steps_per_timestep+1):
                # Calculate u from PID
                robot_step_states = self.model_states_from_state_tensor(
                    step_states[..., step_idx - 1, :]
                )[robot_model_name + "_state"]
                assert robot_step_states.size() == batch_dims + (robot_space.n_x,)
                interp_val = (step_idx - 1.0) / steps_per_timestep
                robot_target_states = (
                    interp_val * robot_target_trajectories[..., sim_idx - 1, :]
                    + (1.0 - interp_val) * robot_target_trajectories[..., sim_idx, :]
                )
                assert robot_target_states.size() == batch_dims + (robot_space.n_x,)
                step_u = kp * (
                    robot_space.q(robot_target_states)
                    - robot_space.q(robot_step_states)
                ) + kd * (
                    robot_space.v(robot_target_states)
                    - robot_space.v(robot_step_states)
                )
                # Run Forward Dynamics
                step_q = self.space.q(step_states[..., step_idx - 1, :]).clone()
                step_v = self.space.v(step_states[..., step_idx - 1, :]).clone()
                step_vplus, step_contact_forces, step_contact_normals = self.forward_dynamics(
                    step_q, step_v, step_u, step_dt
                )
                step_states[..., step_idx, :] = self.space.x(
                    self.space.euler_step(step_q, step_v, step_dt), step_vplus
                )
                for key in step_contact_forces:
                    if key not in ret_contact_forces.keys():
                        ret_contact_forces[key] = torch.zeros(batch_dims + (traj_len-1, 3))
                    ret_contact_forces[key][..., sim_idx-1, :] += step_contact_forces[key][..., 0, :]
                for key in step_contact_normals:
                    if key not in ret_contact_normals.keys():
                        ret_contact_normals[key] = torch.zeros(batch_dims + (traj_len-1, 3))
                    ret_contact_normals[key][..., sim_idx-1, :] = step_contact_forces[key][..., 0, :]    
                ret_u[..., sim_idx, :] += step_u
            for key in ret_contact_forces:
                ret_contact_forces[key][..., sim_idx-1, :] /= steps_per_timestep
            ret_u[..., sim_idx, :] /= steps_per_timestep
            plant_states[..., sim_idx, :] = step_states[..., -1, :]

        ret = self.model_states_from_state_tensor(plant_states)
        return ret, ret_contact_forces, ret_contact_normals, ret_u

    def exploration_parameters(self) -> Iterable[Parameter]:
        """
        Parameters specifically used for exploration
        """
        return chain([self._trajectory.current_pose_param()],
            self.multibody_terms.parameters()
        )

    @torch.no_grad
    def get_learned_pose(self) -> Tensor:
        """ Current pose for the learned object """
        return self._trajectory.current_pose_param().detach().clone()

    @torch.no_grad
    def get_learned_trajectory(self) -> Tensor:
        """ Current pose trqjectory for the learned object """
        return self._trajectory.get_current_pose_traj()

    @torch.no_grad
    def get_learned_geometry(self) -> Shape:
        """ Current geometry as a Drake Shape. """
        assert len(self._trajectory_model_names) == 1, "Only 1 learnable object supported"
        model_name = self._trajectory_model_names[0]
        plant = self.plant_diagram.plant
        bodies = get_bodies_in_model_instance(plant, plant.GetModelInstanceByName(model_name))
        assert len(bodies) == 1, "Only 1 learnable body supported"
        body_id = unique_body_identifier(plant, bodies[0])
        body_geometry_indices = self.multibody_terms.geometry_body_assignment[body_id]
        assert len(body_geometry_indices) == 1, "Only 1 learnable geometry"
        body_geometry = cast(CollisionGeometry, self.multibody_terms.contact_terms.geometries[body_geometry_indices[0]])
        return PydrakeToCollisionGeometryFactory.reverse_convert(body_geometry)

    @torch.no_grad
    def get_body_geometry(self, body_name: str) -> Shape:
        """Current geometry of body based on name"""
        plant = self.plant_diagram.plant
        body = plant.GetBodyByName(body_name)
        body_id = unique_body_identifier(plant, body)
        body_geometry_indices = self.multibody_terms.geometry_body_assignment[body_id]
        assert len(body_geometry_indices) == 1, "Body must contain only 1 geometry"
        body_geometry = cast(CollisionGeometry, self.multibody_terms.contact_terms.geometries[body_geometry_indices[0]])
        return PydrakeToCollisionGeometryFactory.reverse_convert(body_geometry)

    @gin.configurable
    def observed_info(
        self,
        data: Optional[DataLoader],
        get_loss_args: Callable[[Tensor, Tensor, MultibodyLearnableSystem], Tensor],
        use_hessian: bool = False
    ) -> Tensor:
        """
        Calculate the Observed Information in previously taken actions

        This is the Hessian of the Loss function at the current estimate
        given past data.

        Use: https://stackoverflow.com/questions/64997817/how-to-compute-hessian-of-the-loss-w-r-t-the-parameters-in-pytorch-using-autogr
        """
        n_params = len(torch.cat([param.flatten() for param in self.exploration_parameters() if param.requires_grad]))
        # TODO: Make this a hyperparam
        ret = 1e-2 * torch.eye(n_params) #torch.zeros((n_params, n_params))
        if data is None:
            return ret

        losses = []
        for xy_i in data:
            x_past: Tensor = xy_i[0]
            x_plus: Tensor = xy_i[1]

            loss = self.contactnets_loss(**get_loss_args(x_past, x_plus, self))
            losses.append(loss)

        # Compute Epoch Average
        all_losses = torch.cat(losses)
        param_list = [param for param in self.exploration_parameters() if param.requires_grad]

        if use_hessian:
            grads = torch.autograd.grad(all_losses.mean(), param_list, retain_graph=True, create_graph=True)
            flattened_list = [grad.flatten() for grad in grads]
            flattened_grads = torch.cat(flattened_list)
            assert len(flattened_grads) == n_params
            hessian = torch.autograd.grad(flattened_grads, param_list, grad_outputs=torch.eye(n_params), is_grads_batched=True, retain_graph=True)
            ret += torch.cat([hess.reshape((n_params, -1)) for hess in hessian], dim=-1)
        else:
            grads = torch.autograd.grad(all_losses, param_list, grad_outputs=torch.eye(all_losses.numel()), is_grads_batched=True)
            grads_tensor = torch.cat([grad.reshape((all_losses.numel(), -1)) for grad in grads], dim=-1)
            assert grads_tensor.size() == (all_losses.numel(), n_params)
            # Compute Fisher Infos as outer product
            per_timestep_fishers = pbmm(grads_tensor.unsqueeze(-1), grads_tensor.unsqueeze(-2))
            summed_fishers = per_timestep_fishers.sum(dim=0)
            assert summed_fishers.size() == ret.size()
            ret += summed_fishers
        try:
            assert not torch.any(torch.isnan(ret))
        except AssertionError:
            breakpoint()
        return ret

    def expected_fisher_info(
        self,
        robot_trajectories: Tensor,
        robot_timestamps: Tensor,
        robot_model_name: str,
    ) -> Tensor:
        """
        Calculate the trace of the fisher information matrix for each robot action.

        Args:
            robot_trajectories: Tensor (batch, traj_len, robot n_x)
            robot_timestamps: Tensor (traj_len,)

        Returns:
            Fisher Information Trace: Tensor (batch, n_params, n_params)
        """
        n_samples = self._hyperparameters.n_fisher_samples
        # Input Validation
        assert len(robot_trajectories.size()) >= 3
        batch_dims = robot_trajectories.size()[:-2]
        traj_len = robot_trajectories.size()[-2]
        assert robot_trajectories.size() == batch_dims + (
            traj_len,
            self._model_spaces[robot_model_name].n_x,
        )
        assert robot_timestamps.size() == (traj_len,)
        # Zero Gradient before Simulation
        self.zero_grad()

        # Simulate batch of robot actions
        # TODO: HACK don't hardcode this
        steps_per_timestep = 1
        plant_states_dict, contact_forces_star, contact_normals_star, robot_u = self.diff_simulate(
            robot_trajectories, robot_timestamps, robot_model_name, 
            kp=20.,
            kd=10.,
            steps_per_timestep=steps_per_timestep
        )
        plant_states = super().construct_state_tensor(plant_states_dict)
        plant_x = plant_states[..., : traj_len - 1, :]
        plant_xplus = plant_states[..., 1:, :]
        robot_u_cropped = robot_u[..., : traj_len - 1, :]
        assert plant_x.size() == plant_xplus.size()

        # Sample forces
        # TODO: HACK contact_forces_star only includes 1:1 collisions, which is all we want
        forces_std = 0.01 # 10g * g ~ 0.01N
        samplers_forces = {}
        for key in contact_forces_star.keys():  
            samplers_forces[key] = Normal(
                loc=contact_forces_star[key].flatten(),
                scale=forces_std * torch.ones_like(contact_forces_star[key].flatten()),
            )
        
        n_params = len(torch.cat([param.flatten() for param in self.exploration_parameters() if param.requires_grad]))
        param_list = [param for param in self.exploration_parameters() if param.requires_grad]
        ret = torch.zeros(batch_dims + (n_params, n_params))

        sample_fishers = []
        loss_batches = torch.zeros(batch_dims + (n_samples,))
        for sample_idx in range(-1, n_samples):
            print(f"Calculate Loss for Sample {sample_idx+1} / {n_samples}...")
            # Impulses need to be a column vector
            sample_contact_forces = {}
            for key in contact_forces_star.keys():
                if sample_idx < 0:
                    sample_contact_forces[key] = contact_forces_star[key]
                else:
                    sample_contact_forces[key] = samplers_forces[key].sample().reshape(contact_forces_star[key].size())

            # Compute Loss (i.e. log-likelihood)
            loss_trajlen_batch = self.contactnets_loss(
                plant_x, robot_u_cropped, plant_xplus, contact_normals=contact_normals_star, contact_forces=sample_contact_forces
            )
            assert loss_trajlen_batch.size() == batch_dims + (traj_len-1,)
            if sample_idx >= 0:
                loss_batches[..., sample_idx] = torch.sum(loss_trajlen_batch, dim=-1).flatten()
            else:
                breakpoint()
        breakpoint()
        # Compute Gradients (i.e. score)
        # TODO: Trade-Off Between Time and VRAM
        sample_fishers =torch.zeros(batch_dims + (n_samples, n_params, n_params))
        for sample_idx in range(-1, n_samples):
            print(f"Computing Gradients for sample {sample_idx+1}/{n_samples}...")
            grads = torch.autograd.grad(loss_batches[..., sample_idx].flatten(), param_list, grad_outputs=torch.eye(loss_batches[..., sample_idx].numel()), is_grads_batched=True, retain_graph=True)
            score_batches = torch.cat([grad.reshape((loss_batches[..., sample_idx].numel(), -1)) for grad in grads], dim=-1)
            assert score_batches.size() == (loss_batches[..., sample_idx].numel(), n_params)
            # Compute Fisher Info as outer product
            if sample_idx >= 0:
                sample_fishers[..., sample_idx, :, :] = pbmm(score_batches.unsqueeze(-1), score_batches.unsqueeze(-2)).reshape(batch_dims + (n_params, n_params))
            else:
                breakpoint()
        breakpoint()
        ret = sample_fishers.sum(dim=-3)
        ret /= n_samples
        # clear gradients
        self.zero_grad()
        return ret

    def add_trajectories(
        self, traj_lens: List[int], traj_data: Optional[List[Optional[Tensor]]] = None
    ):
        """Add new learnable trajectory of length"""
        if traj_data is None:
            traj_data = [None] * len(traj_lens)
        assert len(traj_data) == len(traj_lens)
        for traj_len, traj_datum in zip(traj_lens, traj_data):
            self._trajectory.add_trajectory(traj_len, traj_datum)

    def construct_state_tensor(
        self, data_state: Tensor, state_key: Optional[str] = None
    ) -> Tensor:
        """Input:
        data_state: Tensor coming from the TrajectorySet Dataloader,
                    this class expects a TensorDict, shape [batch, ?]
        state_key: if set, override return with this key
        Returns: full state tensor (adding traj parameters) shape [batch, n_x_full]
        """
        if state_key is not None and state_key in data_state:
            return data_state[state_key]

        # Don't mutate input
        assert isinstance(data_state, TensorDictBase)
        data_state = data_state.clone()

        # Inject trajectory_model's state if not already present
        if "traj_num" in data_state and "index" in data_state:
            assert data_state["traj_num"].numel() == data_state.numel()
            assert data_state["index"].numel() == data_state.numel()
            traj_states = self._trajectory(
                data_state["traj_num"].squeeze(-1), data_state["index"].squeeze(-1)
            )
            traj_splits = self._trajectory.space.x_split(traj_states)
            for traj_model_idx, traj_model_name in enumerate(
                self._trajectory_model_names
            ):
                if (traj_model_name + "_state") not in data_state:
                    model_x = traj_splits[traj_model_idx]
                    data_state[traj_model_name + "_state"] = model_x

        # Return full state using DrakeSystem's function
        return super().construct_state_tensor(data_state)
