import jax
import jax.numpy as jnp
from typing import Tuple

def project_second_order_cone_jax(l: jnp.ndarray) -> jnp.ndarray:
    """
    Projects a batch of vectors onto a series of 3D Lorentz (second-order) cones.
    
    Expected shape of l: (n_batch, 3 * k)
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
    safe_norm_xy = jnp.maximum(norm_xy, 1e-12)
    scale = 0.5 * (1.0 + z / safe_norm_xy)
    proj_xy = xy * scale
    proj_z = norm_xy * scale
    
    proj_triplet = jnp.concatenate([proj_xy, proj_z], axis=-1)
    
    # Vectorized conditional application
    out = jnp.where(in_cone, l_reshaped, proj_triplet)
    out = jnp.where(in_polar, jnp.zeros_like(l_reshaped), out)
    
    return out.reshape(l.shape)

def _mvp_jax_batched(J: jnp.ndarray, v: jnp.ndarray, eps: float) -> jnp.ndarray:
    """
    Batched Matrix-Vector Product H v = (J @ J.T + eps * I) v
    using 3D tensors (n_batch, dim_q, dim_inner).
    """
    # J is (n_batch, dim_q, dim_inner)
    # v is (n_batch, dim_q, 1)
    
    # J.T @ v
    jt_v = jnp.matmul(jnp.swapaxes(J, -1, -2), v)
    # J @ (J.T @ v)
    j_jt_v = jnp.matmul(J, jt_v)
    
    return j_jt_v + eps * v

@jax.jit(static_argnums=(3,))
def accelerated_pgd_socp_sappy_jax(J: jnp.ndarray, q: jnp.ndarray, eps: float, max_iter: int = 100) -> jnp.ndarray:
    """
    Solves SOCP problems using explicit 3D batching and Linear Operators.
    """
    n_batch, dim_q = q.shape
    
    # Estimate the maximum eigenvalue of H via power iteration (Batched)
    v = jax.random.normal(jax.random.PRNGKey(0), (n_batch, dim_q, 1), dtype=jnp.float32)
    
    def power_iter_step(v, _):
        v = _mvp_jax_batched(J, v, eps)
        v = v / jnp.linalg.norm(v, axis=1, keepdims=True)
        return v, None
    
    v, _ = jax.lax.scan(power_iter_step, v, None, length=5)
    
    # Rayleigh quotient
    Hv = _mvp_jax_batched(J, v, eps)
    L = jnp.matmul(jnp.swapaxes(v, -1, -2), Hv).squeeze(-1)
    step_size = 1.0 / jnp.maximum(L, 1e-6)
    
    # Optimization Initialization
    l_init = jnp.zeros((n_batch, dim_q), dtype=jnp.float32)
    y_init = jnp.zeros((n_batch, dim_q), dtype=jnp.float32)
    t_init = 1.0
    
    initial_state = (l_init, l_init, y_init, t_init)
    
    def scan_body(state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float], _):
        l, l_old, y, t = state
        
        # Compute gradient: grad = H * y + q = J @ (J.T @ y) + eps * y + q
        y_unsqueezed = y[..., jnp.newaxis]
        grad = _mvp_jax_batched(J, y_unsqueezed, eps).squeeze(-1) + q
        
        # Descent step and cone projection
        l_next = project_second_order_cone_jax(y - step_size * grad)
        
        # Nesterov momentum update
        t_next = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
        y_next = l_next + ((t - 1.0) / t_next) * (l_next - l)
        
        return (l_next, l, y_next, t_next), None

    final_state, _ = jax.lax.scan(scan_body, initial_state, None, length=max_iter)
    l_final, _, _, _ = final_state
    
    return l_final
