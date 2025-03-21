
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
import time


# class TrajFactory():
#     @staticmethod
#     def initial_uniform_traj():
        
#         """
#         Generates a trajectory
#         HACK: informed by ground truth
#         standardize information gain by enforcing the same action in the initial trial

#         """
        


def collision_free_traj(
    init_state: np.ndarray,
    target_state: np.ndarray,
    constraint_rad: float,
    workspace_rad: float,
    n: int = 100) -> np.ndarray:
    """
    Remember: each row is a waypoint with the number of rows being the steps(# of waypoint)
    """

    target_state = target_state[:9]
    assert workspace_rad > constraint_rad 

    target_dis = np.array([np.linalg.norm(i) for i in [target_state[:3], target_state[3:6], target_state[6:]]])
    init_dis = np.array([np.linalg.norm(i) for i in [init_state[:3], init_state[3:6], init_state[6:]]])
    
    # If the target or init is the inner sphere
    if np.any(target_dis < constraint_rad) or np.any(init_dis < constraint_rad):

        soln = np.linspace(init_state, target_state, num = n)
        soln_dot = np.gradient(soln, axis = 0)

    else:
        # a cost function that minimizes distance in cartesian
        def cost_func(x):
            x_mat = x.reshape((n, 9))
            dx = np.gradient(x_mat, axis = 0)
            return np.sum(dx**2)

        traj_guess = np.random.uniform(0, 1, size = (n, 9))
        
        x0 = traj_guess.flatten()

        # constraint on start point
        start_eq = lambda x: x[:9]
        start_cons = NonlinearConstraint(start_eq, lb = init_state, ub = init_state)

        # constraint on end point
        end_eq = lambda x: x[-9:]
        end_cons = NonlinearConstraint(end_eq, lb = target_state, ub = target_state)

        radial_eq = lambda x: np.array([np.linalg.norm(arr) for arr in np.split(x, n * 3)])
        radial_cons = NonlinearConstraint(radial_eq, lb = constraint_rad, ub = workspace_rad)
        
        result = minimize(cost_func, x0, constraints={start_cons, end_cons, radial_cons})

        soln = result.x.reshape((n, 9))
        assert result.success, "Optimizer failed!"

        soln_dot = np.gradient(soln, axis = 0)

    # No velocity at ends
    soln_dot[0,:] = 0
    soln_dot[-1,:] = 0

    return soln, soln_dot

if __name__ == '__main__':

    collision_free_traj(-np.ones((9,)), np.ones((9,)), steps=10)