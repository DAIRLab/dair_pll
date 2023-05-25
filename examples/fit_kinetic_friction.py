# pylint: disable=E1103
import os
from math import sqrt
from textwrap import dedent

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

from dair_pll import file_utils
from dair_pll.dataset_management import ExperimentDataManager, \
    TrajectorySliceConfig, DataConfig
from dair_pll.multibody_learnable_system import MultibodyLearnableSystem
from dair_pll.tensor_utils import pbmm
from param_error_study import URDFS, DT

STORAGE_NAME = 'cube_kinetic_friction_fit'
STORAGE = os.path.join(os.path.dirname(__file__), 'storage', STORAGE_NAME)
CUBE_DATA_ASSET = 'contactnets_cube'

PHI_MAX_TOL = 1e-3
HORIZONTAL_VELOCITY_MIN = 1e-2
VERTICAL_VELOCITY_MAX = 1e-3
SPIN_AXIS_TOL = .99

LEASTSQ_PRINT_STR = dedent('''
#####################
Kinetic friction fit: Equal-force Least Squares

Number of accepted datapoints: %i
Least squares fit mu: %f
average mu: %f
mu std. dev.: %f
mu std. err.: %f
''')

CVXPY_PRINT_STR = dedent('''
#####################
Kinetic friction fit: Least-squares, acceleration-aligned force split.

Number of accepted datapoints: %i
average mu: %f
mu std. dev.: %f
mu std. err.: %f
''')

BIG_PRINT_STR = dedent('''
#####################
Kinetic friction fit: Large least-squares problem, with constrained equal mu.

Number of accepted datapoints: %i
accpeted mu: %f
squared residuals: %f
''')


def main():
    """Identify the kinetic friction coefficient of the cube.

    We filter datapoints for the conditions that 4 corners are in contact;
    the linear velocity is horizontal; and the angular velocity is vertical.

    We then fit the kinetic friction coefficient using three methods:
        1. Perform a least squares fit to the kinetic friction coefficient.
          This is done by assuming that the weight of the cube is equally
          supported by each corner. The net frictional force direction and
          magnitude are inferred, and compared to the observed deceleration.
        2. For each datapoint, perform a non-negative least squares fit to
          the force apportionment at each corner, assuming their directions
          exactly oppose the sliding direction. The kinetic friction
          coefficient is then inferred from these forces as a scaled sum of
          these coefficients.
        3. Perform one large least squares optimization with both mu and the
          apportionment of the weight to each corner as variables, and report
          the mu directly solved for.
    """
    import_directory = file_utils.get_asset(CUBE_DATA_ASSET)
    file_utils.import_data_to_storage(STORAGE, import_data_dir=import_directory)
    multibody_system = MultibodyLearnableSystem(init_urdfs=URDFS, dt=DT)
    space = multibody_system.space
    integrator = multibody_system.integrator
    data_config = DataConfig(dt=DT,
                             train_valid_test_quantities=(1.0, 0.0, 0.0),
                             slice_config=TrajectorySliceConfig(),
                             use_ground_truth=True)

    data_manager = ExperimentDataManager(STORAGE, data_config)
    train_set, _, _ = data_manager.get_updated_trajectory_sets()
    all_x_0, all_v_f = train_set.slices[:]
    x_0 = torch.cat(all_x_0, dim=0)
    x_f = torch.cat(all_v_f, dim=0)
    q_0 = space.q(x_0)
    v_0 = space.v(x_0)
    u_0 = torch.zeros(q_0.shape[:-1] + (0,))
    q_f = space.q(x_f)
    v_f = space.v(x_f)
    u_f = torch.zeros(q_f.shape[:-1] + (0,))
    _, _, J_0, phi_0, _ = multibody_system.multibody_terms(q_0, v_0, u_0)
    _, _, J_f, phi_f, _ = multibody_system.multibody_terms(q_f, v_f, u_f)
    #pdb.set_trace()
    close_to_ground = (phi_0.max(dim=1).values + phi_f.max(dim=1).values) < \
                      PHI_MAX_TOL
    omega_0 = v_0[:, 0:3]
    v_com_0 = v_0[:, 3:6]
    omega_f = v_f[:, 0:3]
    v_com_f = v_f[:, 3:6]
    is_sliding = (v_com_0[:, 0:2].norm(dim=-1) > HORIZONTAL_VELOCITY_MIN) & \
                 (v_com_f[:, 0:2].norm(dim=-1) > HORIZONTAL_VELOCITY_MIN) & \
                 (v_com_0[:, 2].abs() < VERTICAL_VELOCITY_MAX) & \
                 (v_com_f[:, 2].abs() < VERTICAL_VELOCITY_MAX)
    omega_z_0 = omega_0[:, 2].abs() / omega_0.norm(dim=1)
    omega_z_f = omega_f[:, 2].abs() / omega_f.norm(dim=1)
    is_spinning = (omega_z_0 > SPIN_AXIS_TOL) & (omega_z_f > SPIN_AXIS_TOL)
    is_sliding_example = close_to_ground & is_sliding & is_spinning

    # kinematics
    v_0 = v_0[is_sliding_example]
    v_f = v_f[is_sliding_example]
    J_0 = J_0[is_sliding_example]
    J_f = J_f[is_sliding_example]
    phi_0 = phi_0[is_sliding_example]
    n_accepted = phi_0.shape[0]

    n_contacts = phi_0.shape[-1]
    J_t_0 = J_0[..., n_contacts:, :]
    J_t_f = J_f[..., n_contacts:, :]
    v_t = 0.5 * (pbmm(J_t_0, v_0.unsqueeze(-1)) + \
                 pbmm(J_t_f, v_f.unsqueeze(-1))).reshape(
        phi_0.shape[:-1] + (n_contacts, 2))
    com_horizontal_dv = v_f[:, 3:5] - v_0[:, 3:5]
    s_t = v_t.norm(dim=-1, keepdim=True)
    v_t_hat = v_t / s_t
    sliding_vector_sum_norm = v_t_hat.sum(dim=-2).norm(dim=-1)
    A = sliding_vector_sum_norm * DT * 9.81
    b = n_contacts * com_horizontal_dv.norm(dim=-1)
    mu_fit = torch.linalg.lstsq(A.unsqueeze(-1), b).solution
    individual_mus = b / A
    std = individual_mus.std()
    mean = individual_mus.mean()
    std_err = std / sqrt(individual_mus.shape[0])

    print(LEASTSQ_PRINT_STR % (n_accepted, mu_fit, mean, std, std_err))

    ## individualized solver
    variables = cp.Variable(n_contacts, nonneg=True)
    objective_matrix = cp.Parameter((2, n_contacts))
    objective_vector = cp.Parameter(2)

    objective = cp.sum_squares(objective_matrix @ variables - objective_vector)
    constraints = [variables >= 0.]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    layer = CvxpyLayer(problem,
                       parameters=[objective_matrix, objective_vector],
                       variables=[variables])
    matrices = -v_t_hat.mT * 9.81 * DT
    vectors = com_horizontal_dv
    individual_mus = layer(matrices, vectors)[0].sum(dim=-1)
    std = individual_mus.std()
    mean = individual_mus.mean()
    std_err = std / sqrt(individual_mus.shape[0])
    print(CVXPY_PRINT_STR % (n_accepted, mean, std, std_err))
    force_vars = cp.Variable((n_accepted, n_contacts), nonneg=True)
    mu_var = cp.Variable(1, nonneg=True)
    objective = 0.
    constraints = []
    for i in range(n_accepted):
        mi = matrices[i].detach().numpy()
        vi = vectors[i].detach().numpy()
        objective += cp.sum_squares(mi @ force_vars[i, :] - vi)
        constraints.append(cp.sum(force_vars[i, :]) == mu_var)
    constraints.append(force_vars >= 0.)
    problem = cp.Problem(cp.Minimize(objective), constraints)

    problem.solve()
    print(BIG_PRINT_STR % (n_accepted, mu_var.value, objective.value))


if __name__ == "__main__":
    main()
