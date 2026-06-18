import torch
import time
from dair_pll.socp_solver import accelerated_pgd_socp_sappy

# Try to import JAX-related components
try:
    import jax
    import jax.numpy as jnp
    from dair_pll.socp_jax_solver import accelerated_pgd_socp_sappy_jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

def check_kkt(l, J, q, eps):
    """
    Verifies the KKT conditions for the SOCP problem:
    min 1/2 l^T H l + q^T l subject to l in K
    
    KKT conditions:
    1. Primal feasibility: l in K
    2. Dual feasibility: g = H l + q in K
    3. Complementary slackness: l^T g = 0
    """
    l = l.float()
    J = J.float()
    q = q.float()
    
    is_batched = (l.dim() == 2)
    if not is_batched:
        l = l.unsqueeze(0)
        q = q.unsqueeze(0)
        J = J.unsqueeze(0)
        
    n_batch, dim_q = q.shape
    dim_0, dim_1 = J.shape[1], J.shape[2]
    
    if dim_0 == dim_q:
        H = torch.bmm(J, J.transpose(1, 2))
    elif dim_1 == dim_q:
        H = torch.bmm(J.transpose(1, 2), J)
    else:
        raise ValueError("Dimension mismatch")
        
    eye = torch.eye(dim_q, device=J.device, dtype=J.dtype).unsqueeze(0)
    H = H + eps * eye
    
    # Gradient g = H l + q
    g = torch.bmm(H, l.unsqueeze(-1)).squeeze(-1) + q
    
    # Helper to check if in cone
    def in_cone(v):
        v_reshaped = v.reshape(n_batch, -1, 3)
        xy_norm = torch.norm(v_reshaped[..., :2], p=2, dim=-1)
        z = v_reshaped[..., 2]
        return z - xy_norm

    primal_feas = in_cone(l)
    dual_feas = in_cone(g)
    complementarity = torch.abs(torch.sum(l * g, dim=-1))
    
    return {
        'primal_feas_min': torch.min(primal_feas).item(),
        'dual_feas_min': torch.min(dual_feas).item(),
        'complementarity': torch.max(complementarity).item()
    }

def run_benchmark():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")
    max_iter = 50
    test_cases = [
        # (k, n_batch, eps)
        (10, 1, 1e-3),
        (10, 10, 1e-3),
        (10, 100, 1e-3),
        (10, 1000, 1e-3),
        (10, 10000, 1e-3),
        (100, 1, 1e-3),
        (100, 10, 1e-3),
        (100, 100, 1e-3),
        (100, 1000, 1e-3)]
        
    for k, n_batch, eps in test_cases:
        dim_q = 3 * k
        # Random J of size (n_batch, dim_q, dim_q)
        J = torch.randn(n_batch, dim_q, dim_q, device=device)
        J_warm = torch.randn(n_batch, dim_q, dim_q, device=device)
        q = torch.randn(n_batch, dim_q, device=device)
        q_warm = torch.randn(n_batch, dim_q, device=device)
        
        # Benchmark PGD (dair_pll implementation)
        if device.type =='cuda': torch.cuda.synchronize()
        start_time = time.time()
        l_pgd = accelerated_pgd_socp_sappy(J, q, eps, max_iter=max_iter, tol=0)
        if device.type =='cuda': torch.cuda.synchronize()
        end_time = time.time()
        duration_pgd = end_time - start_time
        kkt_pgd = check_kkt(l_pgd, J, q, eps)
        
        
        print(f"\nCase: k={k}, batch={n_batch}, eps={eps}")
        print(f"  PGD Solver:")
        print(f"    Time: {duration_pgd:.4f}s ({n_batch / (duration_pgd + 1e-6):.2f} samples/s)")
        print(f"    Primal Feas: {kkt_pgd['primal_feas_min']:.2e}, Dual Feas: {kkt_pgd['dual_feas_min']:.2e}, Comp: {kkt_pgd['complementarity']:.2e}")

        # Benchmark JAX Solver
        if JAX_AVAILABLE:
            # Convert to JAX arrays
            J_jax = jnp.array(J.cpu().numpy())
            q_jax = jnp.array(q.cpu().numpy())
            J_jax_warm = jnp.array(J_warm.cpu().numpy())
            q_jax_warm = jnp.array(q_warm.cpu().numpy())
            
            # Warm-up (JIT compilation)
            _ = accelerated_pgd_socp_sappy_jax(J_jax_warm, q_jax_warm, eps, max_iter=max_iter).block_until_ready()
            
            if device.type =='cuda': torch.cuda.synchronize()
            start_time = time.time()
            l_jax_raw = accelerated_pgd_socp_sappy_jax(J_jax, q_jax, eps, max_iter=max_iter).block_until_ready()
            if device.type =='cuda': torch.cuda.synchronize()
            end_time = time.time()
            duration_jax = end_time - start_time
            
            # Convert back to PyTorch for KKT check
            l_jax = torch.from_numpy(jax.device_get(l_jax_raw)).to(device).float()
            kkt_jax = check_kkt(l_jax, J, q, eps)
            
            print(f"  JAX Solver:")
            print(f"    Time: {duration_jax:.4f}s ({n_batch / (duration_jax + 1e-6):.2f} samples/s)")
            print(f"    Primal Feas: {kkt_jax['primal_feas_min']:.2e}, Dual Feas: {kkt_jax['dual_feas_min']:.2e}, Comp: {kkt_jax['complementarity']:.2e}")
        else:
            print(f"  JAX Solver: Not available (JAX not installed)")


if __name__ == "__main__":
    run_benchmark()
