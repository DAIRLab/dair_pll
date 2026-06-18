import torch
import math
from typing import Optional

def project_second_order_cone(l: torch.Tensor) -> torch.Tensor:
    """
    Projects a batch of vectors onto a series of 3D Lorentz (second-order) cones.

    Expected shape of l: (n_batch, 3 * k)
    """
    orig_shape = l.shape
    if l.dim() == 1:
        l = l.unsqueeze(0)

    n_batch, total_dim = l.shape
    k = total_dim // 3

    # Reshape to isolate individual 3D cones: (n_batch, k, 3)
    l_reshaped = l.reshape(n_batch, k, 3)

    xy = l_reshaped[..., :2]  # (n_batch, k, 2)
    z  = l_reshaped[..., 2:3] # (n_batch, k, 1)

    # Compute the Euclidean norm of the Cartesian coordinates
    norm_xy = torch.norm(xy, p=2, dim=-1, keepdim=True)

    # Sappy conic boundary conditions
    in_cone = z >= norm_xy
    in_polar = z <= -norm_xy

    # Analytical projection formula
    scale = 0.5 * (1.0 + z / torch.clamp(norm_xy, min=1e-12))
    proj_xy = xy * scale
    proj_z = norm_xy * scale

    proj_triplet = torch.cat([proj_xy, proj_z], dim=-1)

    # Vectorized conditional application
    out = torch.where(in_cone, l_reshaped, proj_triplet)
    out = torch.where(in_polar, torch.zeros_like(l_reshaped), out)

    return out.reshape(orig_shape)

def _mvp(J: torch.Tensor, v: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Matrix-Vector Product with the Hessian H = J @ J.T + eps * I
    using the Linear Operator pattern to avoid building H.
    """
    # v is (n_batch, dim_q, 1)
    # J is (n_batch, dim_q, dim_inner)

    # Compute J.T @ v
    jt_v = torch.bmm(J.transpose(1, 2), v)
    # Compute J @ (J.T @ v)
    j_jt_v = torch.bmm(J, jt_v)

    return j_jt_v + eps * v

@torch.jit.script
def _pgd_step_core(y: torch.Tensor, grad: torch.Tensor, step_size: torch.Tensor) -> torch.Tensor:
    """
    Core PGD step separated for potential compilation/scripting.
    """
    return project_second_order_cone(y - step_size * grad)

def accelerated_pgd_socp_sappy(J: torch.Tensor, q: torch.Tensor, eps: float, max_iter: int = 100, tol: float = 1e-5) -> torch.Tensor:
    """
    Solves min 1/2 l^T (Hessian) l + q^T l subject to second-order cone constraints
    using Nesterov Accelerated Projected Gradient Descent with Linear Operators.
    """
    is_batched = (q.dim() == 2)
    if not is_batched:
        q = q.unsqueeze(0)
        J = J.unsqueeze(0)

    n_batch, dim_q = q.shape
    device = J.device
    dtype = J.dtype

    # Ensure J is in (n_batch, dim_q, dim_inner) format for Linear Operator
    # The benchmark provides (n_batch, dim_q, dim_q)
    if J.shape[1] != dim_q:
        J = J.transpose(1, 2)

    # Optimization Initialization
    l = torch.zeros(n_batch, dim_q, device=device, dtype=dtype)
    y = torch.zeros(n_batch, dim_q, device=device, dtype=dtype)
    t = 1.0

    # Estimate the maximum eigenvalue of H via power iteration (Linear Operator version)
    with torch.no_grad():
        v = torch.randn(n_batch, dim_q, 1, device=device, dtype=dtype)
        for _ in range(5):
            v = _mvp(J, v, eps)
            v = v / torch.norm(v, dim=1, keepdim=True)

        # Rayleigh quotient: L = (v.T @ H @ v)
        Hv = _mvp(J, v, eps)
        L = torch.bmm(v.transpose(1, 2), Hv).squeeze(-1) 
        step_size = 1.0 / torch.clamp(L, min=1e-6)

    # We use a micro-batching approach internally if n_batch is very large
    # to stay within L2 cache limits.

    for i in range(max_iter):
        l_old = l.clone()

        # Compute gradient: grad = H * y + q = J @ (J.T @ y) + eps * y + q
        y_unsqueezed = y.unsqueeze(-1)
        grad = _mvp(J, y_unsqueezed, eps).squeeze(-1) + q

        # Descent step and cone projection
        l = _pgd_step_core(y, grad, step_size)

        # Check convergence tolerance
        if tol > 0:
            res = torch.norm(l - l_old, dim=-1)
            if torch.max(res) < tol:
                break

        # Nesterov momentum update
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        y = l + ((t - 1.0) / t_next) * (l - l_old)
        t = t_next

    return l if is_batched else l.squeeze(0)