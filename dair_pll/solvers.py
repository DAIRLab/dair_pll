"""Convex optimization solver interfaces.
Current supported problem/solver types:
    * Lorentz cone constrained quadratic program (LCQP) solved with CVXPY.
"""

from typing import Any, Optional, Dict, List, cast

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import gin
import numpy as np
import torch
from torch import Tensor
from dair_pll.tensor_utils import sqrtm

_CVXPY_LCQP_EPS = 0.0  # 1e-7


def construct_cvxpy_lcqp_layer(num_contacts: int) -> CvxpyLayer:
    """Constructs a CvxpyLayer for solving a Lorentz cone constrained quadratic
    program.
    Args:
        num_contacts: number of contacts to be considered in the LCQP.
    Returns:
        CvxpyLayer for solving a LCQP.
    """
    num_variables = 3 * num_contacts

    variables = cp.Variable(num_variables)
    objective_matrix = cp.Parameter((num_variables, num_variables))
    objective_vector = cp.Parameter(num_variables)

    objective = 0.5 * cp.sum_squares(objective_matrix @ variables)
    objective += objective_vector.T @ variables
    if _CVXPY_LCQP_EPS > 0.0:
        objective += 0.5 * _CVXPY_LCQP_EPS * cp.sum_squares(variables)
    constraints = [
        cp.SOC(variables[3 * i + 2], variables[(3 * i) : (3 * i + 2)])
        for i in range(num_contacts)
    ]

    problem = cp.Problem(cp.Minimize(objective), cast(List[cp.Constraint], constraints))
    return CvxpyLayer(
        problem, parameters=[objective_matrix, objective_vector], variables=[variables]
    )


# TODO: clean up
# Disable JAX vram hogging
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax
from jaxopt import OSQP
from jax2torch import jax2torch


# TODO: edit jax2torch.py to remove to_dlpack calls
# TODO: edit jax2torch.py to call .detach in t2j()
@jax2torch
@jax.jit
def jaxopt_qp_run(
    Qj: jax.Array, qj: jax.Array, Gj: jax.Array, hj: jax.Array
) -> jax.Array:
    return OSQP().run(params_obj=(Qj, qj), params_ineq=(Gj, hj)).params.primal

@jax2torch
@jax.jit
@jax.vmap
def jaxopt_qp_run_jax_vmap(
    Qj: jax.Array, qj: jax.Array, Gj: jax.Array, hj: jax.Array
) -> jax.Array:
    return OSQP().run(params_obj=(Qj, qj), params_ineq=(Gj, hj)).params.primal


def jaxopt_solver(Qin: Tensor, qin: Tensor, torch_vmap=False) -> Tensor:
    """Solver using jaxopt and the pyramid approximation
    Args:
        Q: (*, 3 * num_contacts, 3 * num_contacts) Cost matrices.
        q: (*, 3 * num_contacts) Cost vectors.
    Returns:
        LCQP solution impulses.
    """
    # Input validation
    assert Qin.shape[-1] % 3 == 0
    assert Qin.shape[-2] == Qin.shape[-1]
    n_c = Qin.shape[-1] // 3
    assert qin.shape[-1] == 3 * n_c
    # Map pyramid space to (l_tx, l_ty, l_n)
    lamb_map = np.cos(np.pi / 4) * np.array(
        [[1, 0, -1, 0], [0, 1, 0, -1], [1, 1, 1, 1]]
    )
    lamb_map_full = torch.tensor(np.kron(np.eye(n_c), lamb_map), dtype=Qin.dtype)

    # Wrap Q and q
    Q_solve = lamb_map_full.T @ Qin.reshape((-1,) + Qin.size()[-2:]) @ lamb_map_full
    q_solve = qin.reshape((-1,) + qin.size()[-1:]) @ lamb_map_full
    Gt = -1.0 * torch.eye(4 * n_c).unsqueeze(0).expand(Q_solve.shape[0], -1, -1)
    ht = torch.zeros(Q_solve.shape[0], 4 * n_c)
    if torch_vmap:
        sol = torch.vmap(jaxopt_qp_run)(Q_solve, q_solve, Gt, ht)
    else:
        sol = jaxopt_qp_run_jax_vmap(Q_solve, q_solve, Gt, ht)
    return (sol.type(lamb_map_full.dtype) @ lamb_map_full.T).reshape(qin.shape)


@gin.configurable
class DynamicCvxpyLCQPLayer:
    """Solves a LCQP with dynamic sizing by maintaining a family of
    constant-size ``CvxpyLayer`` s."""

    _cvxpy_layers: Dict[int, CvxpyLayer]
    _solver_args: Dict[str, Any]

    def __init__(self, solver_args):
        self._cvxpy_layers = {}
        self._solver_args = solver_args

    def get_sized_layer(self, num_contacts: int) -> CvxpyLayer:
        """Returns a ``CvxpyLayer`` for solving a LCQP with ``num_contacts``
        contacts.
        Args:
            num_contacts: number of contacts to be considered in the LCQP.
        Returns:
            CvxpyLayer for solving a LCQP.
        """
        if num_contacts not in self._cvxpy_layers:
            self._cvxpy_layers[num_contacts] = construct_cvxpy_lcqp_layer(num_contacts)
        return self._cvxpy_layers[num_contacts]

    def __call__(self, Q: Tensor, q: Tensor) -> Tensor:
        """Solve an LCQP.
        Args:
            Q: (*, 3 * num_contacts, 3 * num_contacts) Cost matrices.
            q: (*, 3 * num_contacts) Cost vectors.
        Returns:
            LCQP solution impulses.
        """
        Q_solve = Q.reshape((-1,) + Q.size()[-2:])
        q_solve = q.reshape((-1,) + q.size()[-1:])
        assert Q_solve.shape[-2] % 3 == 0
        assert Q_solve.shape[-1] == Q_solve.shape[-2]
        assert q_solve.shape[-1] == Q_solve.shape[-2]

        layer = self.get_sized_layer(Q_solve.shape[-2] // 3)
        Q_sqrt = sqrtm(Q_solve)
        soln = layer(Q_sqrt, q_solve, solver_args=self._solver_args)[0]
        return soln.reshape(q.size())
