import torch
import math

def project_second_order_cone(l: torch.Tensor) -> torch.Tensor:
    """
    Projects a batch of vectors onto a series of 3D Lorentz (second-order) cones.
    
    Expected shape of l: (n_batch, 3 * k)
    Cones are structured such that for each triplet i:
        l_z_i >= sqrt(l_x_i^2 + l_y_i^2)
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

def accelerated_pgd_socp_sappy(J: torch.Tensor, q: torch.Tensor, eps: float, max_iter: int = 100, tol: float = 1e-5) -> torch.Tensor:
    """
    Solves min 1/2 l^T (Hessian) l + q^T l subject to second-order cone constraints
    using Nesterov Accelerated Projected Gradient Descent.
    
    Uses q's size as the ground truth dimension N, and forms an (N x N) Hessian.
    """
    is_batched = (q.dim() == 2)
    
    if not is_batched:
        q = q.unsqueeze(0)
        J = J.unsqueeze(0)
        
    n_batch, dim_q = q.shape
    
    # Extract dimensions from J (excluding batch dim)
    dim_0, dim_1 = J.shape[1], J.shape[2]
    
    # Explicitly build the Hessian to ensure its shape is exactly (dim_q, dim_q)
    if dim_0 == dim_q and dim_1 == dim_q:
        H = torch.bmm(J, J.transpose(1, 2))  # Square matrix case
    elif dim_0 == dim_q:
        H = torch.bmm(J, J.transpose(1, 2))  # (dim_q, dim_1) x (dim_1, dim_q) -> (dim_q, dim_q)
    elif dim_1 == dim_q:
        H = torch.bmm(J.transpose(1, 2), J)  # (dim_q, dim_0) x (dim_0, dim_q) -> (dim_q, dim_q)
    else:
        raise ValueError(
            f"Dimension mismatch: q has size {dim_q}, but J has shape ({dim_0}, {dim_1}). "
            f"Neither dimension of J matches q."
        )
        
    # Apply regularizer safely using the guaranteed dimension match
    eye = torch.eye(dim_q, device=J.device, dtype=J.dtype).unsqueeze(0)
    H = H + eps * eye 
    
    # Optimization Initialization
    l = torch.zeros(n_batch, dim_q, device=J.device, dtype=J.dtype)
    y = torch.zeros(n_batch, dim_q, device=J.device, dtype=J.dtype)
    t = 1.0
    
    # Estimate the maximum eigenvalue of H via power iteration to establish the step size
    with torch.no_grad():
        v = torch.randn(n_batch, dim_q, 1, device=J.device, dtype=J.dtype)
        for _ in range(5):
            v = torch.bmm(H, v)
            v = v / torch.norm(v, dim=1, keepdim=True)
        L = torch.bmm(torch.bmm(v.transpose(1, 2), H), v).squeeze(-1) 
        step_size = 1.0 / torch.clamp(L, min=1e-6)
        
    for i in range(max_iter):
        l_old = l.clone()
        
        # Compute gradient evaluated at look-ahead parameter y: grad = H * y + q
        grad = torch.bmm(H, y.unsqueeze(-1)).squeeze(-1) + q
        
        # Descent step and cone projection
        l = project_second_order_cone(y - step_size * grad)
        
        # Check convergence tolerance
        res = torch.norm(l - l_old, dim=-1)
        if torch.max(res) < tol:
            break
            
        # Nesterov momentum update
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        y = l + ((t - 1.0) / t_next) * (l - l_old)
        t = t_next
        
    return l if is_batched else l.squeeze(0)