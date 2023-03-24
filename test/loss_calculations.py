"""Test script for comparing loss calculations from PLL and SoPhTER."""
import torch
from torch import Tensor

from dair_pll import file_utils
from dair_pll.dataset_management import DataConfig, DataGenerationConfig
from dair_pll.drake_experiment import DrakeMultibodyLearnableExperiment, \
                                      DrakeSystemConfig, \
                                      MultibodyLearnableSystemConfig, \
                                      MultibodyLosses
from dair_pll.experiment import SupervisedLearningExperimentConfig, \
                                OptimizerConfig, default_epoch_callback
from dair_pll.multibody_learnable_system import MultibodyLearnableSystem
from dair_pll.state_space import UniformSampler


NQ = 7
TRUE_CUBE_URDF = 'contactnets_cube.urdf'
LARGE_CUBE_URDF = 'contactnets_cube_large.urdf'
SYSTEM_NAME = 'cube'


class SoPhTERTensorUtils:
	"""Tensor utilities from SoPhTER."""

	@staticmethod
	def matrix_diag(diagonal: Tensor) -> Tensor:
	    """
	    Written by fmassa at: https://github.com/pytorch/pytorch/issues/12160
	    """
	    N = diagonal.shape[-1]
	    shape = diagonal.shape[:-1] + (N, N)
	    device, dtype = diagonal.device, diagonal.dtype
	    result = torch.zeros(shape, dtype=dtype, device=device)
	    indices = torch.arange(result.numel(), device=device).reshape(shape)
	    indices = indices.diagonal(dim1=-2, dim2=-1)
	    result.view(-1)[indices] = diagonal
	    return result

	@staticmethod
	def veceye(n: int, veclen: int) -> Tensor:
	    """Compute a block diagonal matrix with column vectors of ones as blocks.
	    Example:
	    veceye(3, 2) =
	        tensor([[1., 0., 0.],
	                [1., 0., 0.],
	                [0., 1., 0.],
	                [0., 1., 0.],
	                [0., 0., 1.],
	                [0., 0., 1.]])
	    Args:
	        n: number of columns.
	        veclen: number of ones in each matrix diagonal block.
	    Returns:
	        A (n * veclen) x n matrix.
	    """

	    return torch.eye(n).repeat(1, veclen).reshape(n * veclen, n)

	@staticmethod
	def pad_right(x: Tensor, elem: float, num: int) -> Tensor:
	    """Right pad a batched tensor with an element.
	    Args:
	        x: batch_n x n x m.
	        elem: element to pad with.
	        num: how many columns filled with elem to add.
	    Returns:
	        A batch_n x n x (m + num) tensor. The new elements are all filled with elem.
	    """
	    pad = torch.ones(x.shape[0], x.shape[1], num) * elem
	    return torch.cat((x, pad), dim=2)

	@staticmethod
	def pad_left(x: Tensor, elem: float, num: int) -> Tensor:
	    """Left pad a batched tensor with an element.
	    Args:
	        x: batch_n x n x m.
	        elem: element to pad with.
	        num: how many columns filled with elem to add.
	    Returns:
	        A batch_n x n x (num + m) tensor. The new elements are all filled with elem.
	    """
	    pad = torch.ones(x.shape[0], x.shape[1], num) * elem
	    return torch.cat((pad, x), dim=2)

	@staticmethod
	def pad_top(x: Tensor, elem: float, num: int) -> Tensor:
	    """Top pad a batched tensor with an element.
	    Args:
	        x: batch_n x n x m.
	        elem: element to pad with.
	        num: how many rows filled with elem to add.
	    Returns:
	        A batch_n x (num + n) x m tensor. The new elements are all filled with elem.
	    """
	    pad = torch.ones(x.shape[0], num, x.shape[2]) * elem
	    return torch.cat((pad, x), dim=1)

	@staticmethod
	def pad_bottom(x: Tensor, elem: float, num: int) -> Tensor:
	    """Bottom pad a batched tensor with an element.
	    Args:
	        x: batch_n x n x m.
	        elem: element to pad with.
	        num: how many rows filled with elem to add.
	    Returns:
	        A batch_n x (n + num) x m tensor. The new elements are all filled with elem.
	    """
	    pad = torch.ones(x.shape[0], num, x.shape[2]) * elem
	    return torch.cat((x, pad), dim=1)

	@staticmethod
	def diag_append(x: Tensor, elem: float, num: int) -> Tensor:
	    """Diagonally pad a batched tensor with the identity times an element after the orginal.
	    For each batched matrix, make the new matrix block diagonal with the original matrix in the
	    upper left corner and eye(num) * elem in the bottom right corner.
	    Args:
	        x: batch_n x n x m.
	        elem: element to pad with.
	        num: the size of the identity matrix.
	    Returns:
	        A batch_n x (n + num) x (m + num) tensor. The new elements are all filled with elem.
	    """

	    batch_n = x.shape[0]

	    brblock = torch.eye(num).unsqueeze(0).repeat(batch_n, 1, 1) * elem
	    bottom_zeros = torch.zeros(batch_n, num, x.shape[2])
	    bottom_block = torch.cat((bottom_zeros, brblock), dim=2)

	    x = pad_right(x, 0, num)
	    x = torch.cat((x, bottom_block), dim=1)
	    return x

	@staticmethod
	def diag_prepend(x: Tensor, elem: float, num: int) -> Tensor:
	    """Diagonally pad a batched tensor with the identity times an element before the orginal.
	    For each batched matrix, make the new matrix block diagonal with eye(num) * elem in the
	    upper left corner and the original matrix in the bottom right corner.
	    Args:
	        x: batch_n x n x m.
	        elem: element to pad with.
	        num: the size of the identity matrix.
	    Returns:
	        A batch_n x (num + n) x (num + m) tensor. The new elements are all filled with elem.
	    """
	    batch_n = x.shape[0]

	    tlblock = torch.eye(num).unsqueeze(0).repeat(batch_n, 1, 1) * elem
	    top_zeros = torch.zeros(batch_n, num, x.shape[2])
	    top_block = torch.cat((tlblock, top_zeros), dim=2)

	    x = pad_left(x, 0, num)
	    x = torch.cat((top_block, x), dim=1)
	    return x

	@staticmethod
	def robust_sqrt(out_squared: Tensor, eps = 1e-8) -> Tensor:
	    # TODO: write description
	    out = torch.zeros(out_squared.shape)
	    out_big = out_squared >= eps ** 2
	    out_small = torch.logical_not(out_big)
	    out[out_big] = torch.sqrt(out_squared[out_big])
	    out[out_small] = out_squared[out_small] * 0.5 / eps + 0.5 * eps
	    return out

	@staticmethod
	def block_diag(m):
	    """
	    Make a block diagonal matrix along dim=-3
	    EXAMPLE:
	    block_diag(torch.ones(4,3,2))
	    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
	    Prepend batch dimensions if needed.
	    You can also give a list of matrices.
	    :type m: torch.Tensor, list
	    :rtype: torch.Tensor
	    """
	    if type(m) is list:
	        # Remove Nones from list
	        m = utils.filter_none(m)

	        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

	    d = m.dim()
	    n = m.shape[-3]
	    siz0 = m.shape[:-3]
	    siz1 = m.shape[-2:]
	    m2 = m.unsqueeze(-2)
	    eye = attach_dim(torch.eye(n).unsqueeze(-2), d - 3, 1)
	    return (m2 * eye).reshape(siz0 + torch.Size(torch.tensor(siz1) * n))

	@staticmethod
	def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
	    return v.reshape(
	        torch.Size([1] * n_dim_to_prepend) + v.shape + torch.Size([1] * n_dim_to_append))



def sophter_loss(x: Tensor, u: Tensor, x_plus: Tensor):
    sp = self.system.params
    config = self.config
    poly = self.interaction.poly

    # Get configuration / velocity for polygon
    configuration = x[..., :NQ]  #poly.get_configuration_previous()  # noqa
    configuration_plus = x_plus[..., :NQ]  #poly.get_configuration()  # noqa
    velocity = x[..., NQ:]  #poly.get_velocity_previous()
    velocity_plus = x_plus[..., NQ:]  #poly.get_velocity()
    control = u  #poly.get_control_previous()

    batch_n = self.interaction.batch_n()
    bases_n = self.G_bases.shape[0]
    k = self.interaction.contact_n()

    G = self.compute_G()

    phi = self.interaction.compute_phi_previous()
    phi_plus = self.interaction.compute_phi_history()

    Jn = self.interaction.compute_Jn_previous()
    Jt_tilde = self.interaction.compute_Jt_tilde_previous()
    J_tilde = torch.cat((Jn, Jt_tilde), dim=1)

    E_2 = SoPhTERTensorUtils.veceye(k, 2).unsqueeze(0).repeat(batch_n, 1, 1)

    gamma = self.interaction.compute_gamma_previous()
    f = poly.compute_f_previous(sp)

    M = self.interaction.compute_M_previous()
    M_i = self.interaction.compute_M_i_previous()

    F_data = M.bmm(velocity_plus - f)

    # Optimization variables are lambda_n, lambda_t
    def normal_mat_pad(x): return SoPhTERTensorUtils.diag_prepend(x, 0, k)
    def normal_vec_pad(x): return SoPhTERTensorUtils.pad_left(x, 0, k)

    def tangent_mat_pad(x): return SoPhTERTensorUtils.diag_append(x, 0, 2 * k)
    def tangent_vec_pad(x): return SoPhTERTensorUtils.pad_right(x, 0, 2 * k)


    # lambda_n and phi complementarity
    comp_n_A = config.w_comp_n * SoPhTERTensorUtils.matrix_diag((phi_plus.squeeze(2) ** 2))
    comp_n_A = tangent_mat_pad(comp_n_A)


    # lambda_t and phi complementarity
    phi_expand = E_2.bmm(phi_plus)
    comp_t_A = config.w_comp_t * SoPhTERTensorUtils.matrix_diag((phi_expand.squeeze(2) ** 2))
    comp_t_A = normal_mat_pad(comp_t_A)


    # Match impulse data (multiply by M_i to get scaling)
    # Term of form (M_i gamma^T [Jn, Jt_tilde]^T lambda - M_i F)^2
    match_quad_A = M_i.bmm(gamma.transpose(1, 2)).bmm(J_tilde.transpose(1, 2))
    match_quad_b = M_i.bmm(F_data)
    match_A = config.w_match * match_quad_A.transpose(1, 2).bmm(match_quad_A)
    match_b = config.w_match * (-2) * match_quad_b.transpose(1, 2).bmm(match_quad_A)
    match_c = config.w_match * match_quad_b.transpose(1, 2).bmm(match_quad_b)

    # Friction cone boundary
    sliding_vels = Jt_tilde.bmm(gamma).bmm(velocity_plus)
    cone_normal_mat = SoPhTERTensorUtils.matrix_diag(sliding_vels.squeeze(2)).bmm(E_2)

    sliding_vel_norms = E_2.transpose(1, 2).bmm(sliding_vels.mul(sliding_vels))

    if config.robust_sqrt:
        sliding_vel_norms = E_2.bmm(SoPhTERTensorUtils.robust_sqrt(sliding_vel_norms))
    else:
        sliding_vel_norms = E_2.bmm(torch.sqrt(sliding_vel_norms))

    cone_tangent_mat = SoPhTERTensorUtils.matrix_diag(sliding_vel_norms.squeeze(2))
    cone_mat = torch.cat((cone_normal_mat, cone_tangent_mat), dim=2)

    cone_A = config.w_cone * cone_mat.transpose(1, 2).bmm(cone_mat)


    A = comp_n_A + comp_t_A + match_A + cone_A
    b = match_b
    c = match_c


    try:
        full_sol = self.qcqp(2 * A, b.transpose(1,2), torch.rand(b.transpose(1,2).size()), 1e-10, 10000)
        if torch.any(torch.isnan(full_sol)):
            return [torch.tensor([[[0.0]]])]
    except Exception:
        print('LCQP solve fail')
        return [torch.tensor([[[0.0]]])]

    # sum in case batch_n > 1
    qp_loss = utils.compute_quadratic_loss(A, b, c, full_sol).sum()
    contact_mask = torch.norm(F_data, 2, dim=1).unsqueeze(2) > config.w_contact_threshold
    qp_loss = qp_loss * contact_mask.int()
    qp_loss = qp_loss.sum()
    b_zero = torch.zeros(batch_n, 1, 3 * k)
    c_zero = torch.zeros(batch_n, 1, 1)


    loss_terms = [utils.compute_quadratic_loss(comp_n_A, b_zero, c_zero, full_sol).sum(),
                  utils.compute_quadratic_loss(comp_t_A, b_zero, c_zero, full_sol).sum(),
                  utils.compute_quadratic_loss(match_A, match_b, match_c, full_sol).sum(),
                  utils.compute_quadratic_loss(cone_A, b_zero, c_zero, full_sol).sum()]
    

    regularizers = []

    ##### penalize penetration:
    def phi_penalizer(phi): return torch.sum(torch.clamp(-phi, min=0) ** 2)
    pen_loss = config.w_penetration * phi_penalizer(phi_plus)
    regularizers.append(pen_loss)


    ##### constrain config grad normal:
    # L1 cost constraining phi norms w.r.t configuration to be one
    pos_norms = torch.norm(Jn[:, :, 0:3], dim=2)
    grad_normal_loss = config.w_config_grad_normal * \
        ((pos_norms - torch.ones(pos_norms.shape)) ** 2).sum()
    regularizers.append(grad_normal_loss)

    ##### constrain config grad tangent:
    # L1 cost constraining phi_t norms w.r.t configuration to be one
    # NOTE: THIS IS BROKEN, DOESN'T NEED TO BE UNIT, NEEDS TO BE MU!!
    regularizers.append(torch.tensor(0.0))


    ##### constrain config grad perp:
    # L1 cost constraining phi_t norms perpendicular to phi norms
    if torch.norm(Jt_tilde) == 0.0:
        grad_perp_loss = torch.tensor(0.0)
    else:
        def normalize(vecs: Tensor) -> Tensor:
            norms = vecs.norm(dim=2).unsqueeze(2).repeat(1, 1, 3)
            return vecs / norms
        pos_normals = normalize(Jn[:, :, 0:3])
        pos_normals = pos_normals.repeat(1, 1, 2).reshape(batch_n, k * 2, 3)
        pos_tangents = normalize(Jt_tilde[:, :, 0:3])

        grad_perp_loss = config.w_config_grad_perp * \
            ((pos_normals * pos_tangents).sum(dim=2) ** 2).sum()
    regularizers.append(grad_perp_loss)


    ##### constrain st estimate normal:
    # L2 cost on phi_plus_hat deviating from phi_plus
    phi_plus_hat = phi + sp.dt * Jn.bmm(gamma).bmm(velocity_plus)
    st_pen_loss = config.w_st_estimate_pen * \
        torch.sum(torch.clamp(-phi_plus_hat, min=0) ** 2)  # /batch_n
    regularizers.append(st_pen_loss)

    phi_norm = (torch.norm(phi_plus - phi_plus_hat, dim=1) ** 2).sum()  # /batch_n
    st_normal_loss = config.w_st_estimate_normal * phi_norm
    regularizers.append(st_normal_loss)


    ##### constrain st estimate tangent:
    phi_t = self.interaction.compute_phi_t_previous()
    phi_t_plus = self.interaction.compute_phi_t_history()
    phi_t_plus_hat = phi_t + sp.dt * Jt_tilde.bmm(gamma).bmm(velocity_plus)
    phi_t_norm = (torch.norm(phi_t_plus - phi_t_plus_hat, dim=1) ** 2).sum()
    st_tangent_loss = config.w_st_estimate_tangent * phi_t_norm
    regularizers.append(st_tangent_loss)

    # Penalize second derivative of tangent jacobian
    Jt_tilde_plus = self.interaction.compute_Jt_tilde_history()
    delta_vc = (Jt_tilde_plus - Jt_tilde).bmm(gamma).bmm(velocity_plus)
    vc_norm = (torch.norm(delta_vc, dim=1) ** 2).sum()
    tangent_jac_d2_loss = config.w_tangent_jac_d2 * vc_norm
    regularizers.append(tangent_jac_d2_loss)

    total_loss = qp_loss + 0  # Make new variable by adding 0
    for regularizer in regularizers:
        total_loss = total_loss + regularizer

    return [total_loss, qp_loss] + loss_terms + regularizers


def pll_loss(x: Tensor, u: Tensor, x_plus: Tensor) -> 
	Tuple[Tensor, Tensor, Tensor, Tensor]:
	pass


def get_terms(x_plus: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    q_plus = x_plus[..., :NQ]
    v_plus = x_plus[..., NQ:]

	# Begin loss calculation.
    delassus, M, J, phi, non_contact_acceleration = self.multibody_terms(
        q_plus, v_plus, u)


def create_pll_experiment():
	# Describes the optimizer settings.
    optimizer_config = OptimizerConfig()
    optimizer_config.batch_size.value = 1

    # Describes the ground truth system.
    urdfs = {SYSTEM_NAME: file_utils.get_asset(TRUE_CUBE_URDF)}
    base_config = DrakeSystemConfig(urdfs=urdfs)

    # Describes the learnable system.
    learnable_config = MultibodyLearnableSystemConfig(
        urdfs={SYSTEM_NAME: file_utils.get_asset(LARGE_CUBE_URDF)},
        loss=MultibodyLosses.CONTACTNETS_LOSS,
        inertia_mode=0)

    # Describe data source
    data_generation_config = None
    import_directory = None
    dynamic_updates_from = None
    x_0 = X_0S[system]
    if simulation:
    	pass
    elif real:
        # otherwise, specify directory with [T, n_x] tensor files saved as
        # 0.pt, 1.pt, ...
        # See :mod:`dair_pll.state_space` for state format.
        import_directory = file_utils.get_asset(data_asset)
        print(f'Getting real trajectories from {import_directory}\n')
    else:
        dynamic_updates_from = DYNAMIC_UPDATES_FROM

    # Describes configuration of the data
    data_config = DataConfig(
        storage=storage_name,
        # where to store data
        dt=DT,
        train_fraction=1.0 if dynamic else 0.5,
        valid_fraction=0.0 if dynamic else 0.25,
        test_fraction=0.0 if dynamic else 0.25,
        generation_config=data_generation_config,
        import_directory=import_directory,
        dynamic_updates_from=dynamic_updates_from,
        t_prediction=1 if contactnets else T_PREDICTION,
        n_import=dataset_size if real else None)

    # Combines everything into config for entire experiment.
    experiment_config = SupervisedLearningExperimentConfig(
        base_config=base_config,
        learnable_config=learnable_config,
        optimizer_config=optimizer_config,
        data_config=data_config,
        full_evaluation_period=EPOCHS if dynamic else 1,
        # full_evaluation_samples=dataset_size,  # use all available data for eval
        run_tensorboard=tb,
        gen_videos=videos,
        update_geometry_in_videos=True
    )

    # Makes experiment.
    experiment = DrakeMultibodyLearnableExperiment(experiment_config)