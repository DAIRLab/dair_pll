import jax
import jax.numpy as jnp
from typing import Tuple

def project_second_order_cone_jax(l: jnp.ndarray) -> jnp.ndarray:
    """
    Projects a batch of vectors onto a series of 3D Lorentz (second-order) cones.
    """
    n_batch, total_dim = l.shape
    k = total_dim // 3
    l_reshaped = l.reshape(n_batch, k, 3)
    
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

def _mvp_jax_batched(J: jnp.ndarray, v: jnp.ndarray, eps: float) -> jnp.ndarray:
    """
    Batched Matrix-Vector Product H v = (J @ J.T + eps * I) v.
    """
    # J: (n_batch, dim_q, dim_inner), v: (n_batch, dim_q, 1)
    jt_v = jnp.matmul(jnp.swapaxes(J, -1, -2), v)
    j_jt_v = jnp.matmul(J, jt_v)
    return j_jt_v + eps * v

@jax.jit(static_argnums=(3,))
def accelerated_pgd_socp_sappy_jax(J: jnp.ndarray, q: jnp.ndarray, eps: float, max_iter: int = 100) -> jnp.ndarray:
    """
    Solves SOCP problems using unrolled Python loops to ensure memory reuse.
    """
    n_batch, dim_q = q.shape
    
    # Power Iteration (Unrolled to avoid lax.scan memory issues)
    v = jax.random.normal(jax.random.PRNGKey(0), (n_batch, dim_q, 1), dtype=jnp.float32)
    for _ in range(5):
        v = _mvp_jax_batched(J, v, eps)
        v = v / jnp.linalg.norm(v, axis=1, keepdims=True)
    
    Hv = _mvp_jax_batched(J, v, eps)
    L = jnp.matmul(jnp.swapaxes(v, -1, -2), Hv).squeeze(-1)
    step_size = 1.0 / jnp.maximum(L, 1e-6)
    
    # Initialization
    l = jnp.zeros((n_batch, dim_q), dtype=jnp.float32)
    y = jnp.zeros((n_batch, dim_q), dtype=jnp.float32)
    t = 1.0
    
    # Main Loop (Unrolled for discrete kernel launches and memory reuse)
    for _ in range(max_iter):
        l_old = l
        
        # grad = H * y + q
        y_unsqueezed = y[..., jnp.newaxis]
        grad = _mvp_jax_batched(J, y_unsqueezed, eps).squeeze(-1) + q
        
        # Descent step and projection
        l = project_second_order_cone_jax(y - step_size * grad)
        
        # Nesterov momentum
        t_next = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
        y = l + ((t - 1.0) / t_next) * (l - l_old)
        t = t_next
        
    return l
