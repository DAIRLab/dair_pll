import jax
import jax.numpy as jnp
from typing import Tuple

def project_single(l: jnp.ndarray) -> jnp.ndarray:
    """
    Projects a single vector onto a series of 3D Lorentz (second-order) cones.
    Shape: (3 * k,)
    """
    total_dim = l.shape[0]
    k = total_dim // 3
    l_reshaped = l.reshape(k, 3)
    
    xy = l_reshaped[..., :2]
    z  = l_reshaped[..., 2:3]
    
    norm_xy = jnp.linalg.norm(xy, axis=-1, keepdims=True)
    
    in_cone = z >= norm_xy
    in_polar = z <= -norm_xy
    
    safe_norm_xy = jnp.maximum(norm_xy, 1e-12)
    scale = 0.5 * (1.0 + z / safe_norm_xy)
    proj_triplet = jnp.concatenate([xy * scale, norm_xy * scale], axis=-1)
    
    out = jnp.where(in_cone, l_reshaped, proj_triplet)
    out = jnp.where(in_polar, jnp.zeros_like(l_reshaped), out)
    
    return out.reshape(l.shape)

def solve_single(J: jnp.ndarray, q: jnp.ndarray, eps: float, max_iter: int) -> jnp.ndarray:
    """
    Solves a single SOCP instance using Linear Operators.
    J: (dim_q, dim_inner), q: (dim_q,)
    """
    dim_q = q.shape[0]
    
    # Power Iteration (to establish step size)
    v_init = jax.random.normal(jax.random.PRNGKey(0), (dim_q,), dtype=jnp.float32)
    def pi_body(v, _):
        # H v = J @ (J.T @ v) + eps * v
        v_next = jnp.dot(J, jnp.dot(J.T, v)) + eps * v
        v_next = v_next / jnp.linalg.norm(v_next)
        return v_next, None
        
    v_final, _ = jax.lax.scan(pi_body, v_init, None, length=5)
    
    # Rayleigh quotient
    Hv = jnp.dot(J, jnp.dot(J.T, v_final)) + eps * v_final
    L = jnp.dot(v_final, Hv)
    step_size = 1.0 / jnp.maximum(L, 1e-6)
    
    # Optimization Initialization
    l_init = jnp.zeros((dim_q,), dtype=jnp.float32)
    y_init = jnp.zeros((dim_q,), dtype=jnp.float32)
    t_init = 1.0
    
    initial_state = (l_init, l_init, y_init, t_init)
    
    def scan_body(state, _):
        l, l_old, y, t = state
        
        # grad = H * y + q = J @ (J.T @ y) + eps * y + q
        grad = jnp.dot(J, jnp.dot(J.T, y)) + eps * y + q
        
        # Descent step and cone projection
        l_next = project_single(y - step_size * grad)
        
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
    Solves SOCP problems using vmap + Linear Operators.
    """
    # Use vmap to handle batching, which is more memory efficient than 3D tensor math in lax.scan
    vmapped_solver = jax.vmap(solve_single, in_axes=(0, 0, None, None))
    return vmapped_solver(J, q, eps, max_iter)
