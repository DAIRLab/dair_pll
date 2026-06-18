import jax
import jax.numpy as jnp
from typing import Tuple

def project_second_order_cone_jax_single(l: jnp.ndarray) -> jnp.ndarray:
    """
    Projects a single vector onto a series of 3D Lorentz (second-order) cones.
    
    Expected shape of l: (3 * k,)
    """
    total_dim = l.shape[0]
    k = total_dim // 3
    
    # Reshape to isolate individual 3D cones: (k, 3)
    l_reshaped = l.reshape(k, 3)
    
    xy = l_reshaped[..., :2]  # (k, 2)
    z  = l_reshaped[..., 2:3] # (k, 1)
    
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

def _mvp_jax(J: jnp.ndarray, v: jnp.ndarray, eps: float) -> jnp.ndarray:
    """
    Matrix-Vector Product H v = (J @ J.T + eps * I) v
    using the Linear Operator pattern to avoid building H.
    """
    return jnp.matmul(J, jnp.matmul(J.T, v)) + eps * v

def solve_single_pgd(J: jnp.ndarray, q: jnp.ndarray, eps: float, max_iter: int) -> jnp.ndarray:
    """
    Solves a single SOCP problem instance using Linear Operators.
    """
    dim_q = q.shape[0]
    
    # Estimate the maximum eigenvalue of H via power iteration
    v = jax.random.normal(jax.random.PRNGKey(0), (dim_q,), dtype=jnp.float32)
    def power_iter_step(v, _):
        v = _mvp_jax(J, v, eps)
        v = v / jnp.linalg.norm(v)
        return v, None
    
    v, _ = jax.lax.scan(power_iter_step, v, None, length=5)
    L = jnp.dot(v, _mvp_jax(J, v, eps))
    step_size = 1.0 / jnp.maximum(L, 1e-6)
    
    # Optimization Initialization
    l_init = jnp.zeros((dim_q,), dtype=jnp.float32)
    y_init = jnp.zeros((dim_q,), dtype=jnp.float32)
    t_init = 1.0
    
    initial_state = (l_init, l_init, y_init, t_init)
    
    def scan_body(state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float], _):
        l, l_old, y, t = state
        
        # Compute gradient: grad = H * y + q = J @ (J.T @ y) + eps * y + q
        grad = _mvp_jax(J, y, eps) + q
        
        # Descent step and cone projection
        l_next = project_second_order_cone_jax_single(y - step_size * grad)
        
        # Nesterov momentum update
        t_next = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
        y_next = l_next + ((t - 1.0) / t_next) * (l_next - l)
        
        return (l_next, l, y_next, t_next), None

    final_state, _ = jax.lax.scan(scan_body, initial_state, None, length=max_iter)
    l_final, _, _, _ = final_state
    
    return l_final

@jax.jit(static_argnums=(3,))
def accelerated_pgd_socp_sappy_jax(J: jnp.ndarray, q: jnp.ndarray, eps: float, max_iter: int = 100) -> jnp.ndarray:
    """
    Solves SOCP problems using vmap for batching and Linear Operators for efficiency.
    """
    vmapped_solver = jax.vmap(solve_single_pgd, in_axes=(0, 0, None, None))
    return vmapped_solver(J, q, eps, max_iter)
