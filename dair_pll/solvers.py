"""Convex optimization solver interfaces.
Current supported problem/solver types:
    * Lorentz cone constrained quadratic program (LCQP) solved with CVXPY.
"""
from typing import Optional, Dict, List, cast

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
from torch import Tensor
from dair_pll.tensor_utils import sqrtm

_CVXPY_LCQP_EPS = 0.  #1e-7
#_CVXPY_SOLVER_ARGS = {"solve_method": "SCS", "eps": 1e-10, "use_indirect":
# True}
# NOTE: It's faster to do serial since the solve is so quick
# TODO: HACK Recommended to comment out "pre-compute quantities for the derivative" in cone_program.py in diffcp since we don't use it.
_CVXPY_SOLVER_ARGS = {"solve_method": "ECOS", "max_iters": 300,
                      "abstol": 1e-10, "reltol": 1e-10, "feastol": 1e-10,
                      "n_jobs_forward": 1, "n_jobs_backward": 1}

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
    if _CVXPY_LCQP_EPS > 0.:
        objective += 0.5 * _CVXPY_LCQP_EPS * cp.sum_squares(variables)
    constraints = [
        cp.SOC(variables[3 * i + 2], variables[(3 * i):(3 * i + 2)])
        for i in range(num_contacts)
    ]

    problem = cp.Problem(cp.Minimize(objective),
                         cast(List[cp.Constraint], constraints))
    return CvxpyLayer(problem,
                      parameters=[objective_matrix, objective_vector],
                      variables=[variables])


class DynamicCvxpyLCQPLayer:
    """Solves a LCQP with dynamic sizing by maintaining a family of
    constant-size ``CvxpyLayer`` s."""
    num_velocities: int
    _cvxpy_layers: Dict[int, CvxpyLayer]

    def __init__(self):
        self._cvxpy_layers = {}

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
            J: (*, 3 * num_contacts, num_velocities) Cost matrices.
            q: (*, 3 * num_contacts) Cost vectors.
        Returns:
            LCQP solution impulses.
        """
        assert Q.shape[-2] % 3 == 0
        assert Q.shape[-1] == Q.shape[-2]
        assert q.shape[-1] == Q.shape[-2]

        layer = self.get_sized_layer(Q.shape[-2] // 3)
        Q_sqrt = sqrtm(Q)
        return layer(Q_sqrt, q, solver_args=_CVXPY_SOLVER_ARGS)[0]
        