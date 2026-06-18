import jax
import jax.numpy as jnp
from typing import Tuple

def project_second_order_cone_jax(l: jnp.ndarray) -> jnp.ndarray:
    """
    Projects a batch of vectors onto a series of 3D Lorentz (second-order) cones.
    
    Expected shape of l: (n_batch, 3 * k)
    Cones are structured such that for each triplet i:
        l_z_i >= sqrt(l_x_i^2 + l_y_i^2)
    """
    n_batch, total_dim = l.shape
    k = total_dim // 3
    
    # Reshape to isolate individual 3D cones: (n_batch, k, 3)
    l_reshaped = l.reshape(n_batch, k, 3)
    
    xy = l_reshaped[..., :2]  # (n_batch, k, 2)
    z  = l_reshaped[..., 2:3] # (n_batch, k, 1)
    
    # Compute the Euclidean norm of the Cartesian coordinates
    norm_xy = jnp.linalg.norm(xy, axis=-1, keepdims=True)
    
    # Conic boundary conditions
    in_cone = z >= norm_xy
    in_polar = z <= -norm_xy
    
    # Analytical projection formula
    # We use a small epsilon to avoid division by zero
    safe_norm_xy = jnp.maximum(norm_xy, 1e-12)
    scale = 0.5 * (1.0 + z / safe_norm_xy)
    proj_xy = xy * scale
    proj_z = norm_xy * scale
    
    proj_triplet = jnp.concatenate([proj_xy, proj_z], axis=-1)
    
    # Vectorized conditional application
    out = jnp.where(in_cone, l_reshaped, proj_triplet)
    out = jnp.where(in_polar, jnp.zeros_like(l_reshaped), out)
    
    return out.reshape(l.shape)

@jax.jit(static_argnums=(3,))
def accelerated_pgd_socp_sappy_jax(J: jnp.ndarray, q: jnp.ndarray, eps: float, max_iter: int = 100) -> jnp.ndarray:
    """
    Solves min 1/2 l^T (Hessian) l + q^T l subject to second-order cone constraints
    using Nesterov Accelerated Projected Gradient Descent with fixed iterations.
    
    Uses jax.lax.scan for efficient GPU execution via XLA.
    """
    n_batch, dim_q = q.shape
    
    # Build the Hessian: H = J @ J.T + eps * I
    # Assuming J is (n_batch, dim_q, dim_q) as per the benchmark
    H = jnp.matmul(J, jnp.swapaxes(J, -1, -2))
    H = H + eps * jnp.eye(dim_q)
    
    # Estimate the maximum eigenvalue of H via power iteration
    # Use a fixed number of steps for JIT compatibility
    v = jax.random.normal(jax.random.PRNGKey(0), (n_batch, dim_q, 1), dtype=jnp.float32)
    def power_iter_step(v, _):
        v = jnp.matmul(H, v)
        v = v / jnp.linalg.norm(v, axis=1, keepdims=True)
        return v, None
    
    v, _ = jax.lax.scan(power_iter_step, v, None, length=5)
    L = jnp.matmul(jnp.matmul(jnp.swapaxes(v, -1, -2), H), v).squeeze(-1)
    step_size = 1.0 / jnp.maximum(L, 1e-6)
    
    # Optimization Initialization
    l_init = jnp.zeros((n_batch, dim_q), dtype=jnp.float32)
    y_init = jnp.zeros((n_batch, dim_q), dtype=jnp.float32)
    t_init = 1.0
    
    initial_state = (l_init, l_init, y_init, t_init)
    
    def scan_body(state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float], _):
        l, l_old, y, t = state
        
        # Compute gradient evaluated at look-ahead parameter y: grad = H * y + q
        grad = jnp.matmul(H, y[..., jnp.newaxis]).squeeze(-1) + q
        
        # Descent step and cone projection
        l_next = project_second_order_cone_jax(y - step_size * grad)
        
        # Nesterov momentum update
        t_next = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
        y_next = l_next + ((t - 1.0) / t_next) * (l_next - l)
        
        return (l_next, l, y_next, t_next), None

    final_state, _ = jax.lax.scan(scan_body, initial_state, None, length=max_iter)
    l_final, _, _, _ = final_state
    
    return l_final
