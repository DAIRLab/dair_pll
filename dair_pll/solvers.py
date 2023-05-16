"""Convex optimization solver interfaces.

Current supported problem/solver types:

    * Lorentz cone constrained quadratic program (LCQP) solved with CVXPY.

"""
from typing import Dict, List, cast

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from torch import Tensor

_CVXPY_LCQP_EPS = 0.  #1e-7
#_CVXPY_SOLVER_ARGS = {"solve_method": "SCS", "eps": 1e-10, "use_indirect":
# True}
_CVXPY_SOLVER_ARGS = {"solve_method": "ECOS", "max_iters": 300,
                      "abstol": 1e-10, "reltol": 1e-10, "feastol": 1e-10}

def construct_cvxpy_lcqp_layer(num_contacts: int,
                               num_velocities: int) -> CvxpyLayer:
    """Constructs a CvxpyLayer for solving a Lorentz cone constrained quadratic
    program.

    Args:
        num_contacts: number of contacts to be considered in the LCQP.
        num_velocities: number of generalized velocities.

    Returns:
        CvxpyLayer for solving a LCQP.
    """
    num_variables = 3 * num_contacts

    variables = cp.Variable(num_variables)
    objective_matrix = cp.Parameter((num_variables, num_velocities))
    objective_vector = cp.Parameter(num_variables)

    objective = 0.5 * cp.sum_squares(objective_matrix.T @ variables)
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

    def __init__(self, num_velocities: int):
        """
        Args:
            num_velocities: number of generalized velocities.
        """
        self.num_velocities = num_velocities
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
            self._cvxpy_layers[num_contacts] = construct_cvxpy_lcqp_layer(
                num_contacts, self.num_velocities)
        return self._cvxpy_layers[num_contacts]

    def __call__(self, J: Tensor, q: Tensor) -> Tensor:
        """Solve an LCQP.

        Args:
            J: (*, 3 * num_contacts, num_velocities) Cost matrices.
            q: (*, 3 * num_contacts) Cost vectors.

        Returns:
            LCQP solution impulses.
        """
        assert J.shape[-1] == self.num_velocities
        assert q.shape[-1] == J.shape[-2]
        assert J.shape[-2] % 3 == 0

        layer = self.get_sized_layer(J.shape[-2] // 3)
        #pdb.set_trace()
        return layer(J, q, solver_args=_CVXPY_SOLVER_ARGS)[0]
