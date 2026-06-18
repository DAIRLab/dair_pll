import torch
import jax
import jax.numpy as jnp
from dair_pll.socp_solver import accelerated_pgd_socp_sappy
from dair_pll.socp_jax_solver import accelerated_pgd_socp_sappy_jax
from sappy import SAPSolver

def calculate_objective(l, J, q, eps):
    """Calculates 1/2 l^T (J J^T + eps I) l + q^T l"""
    l = l.float()
    J = J.float()
    q = q.float()
    
    dim_q = q.shape[-1]
    H = (torch.bmm(J, J.transpose(1, 2)) + eps * torch.eye(dim_q, device=J.device)).float()
    
    # Ensure l has batch dimension for bmm
    if l.dim() == 1:
        l = l.unsqueeze(0)
    if q.dim() == 1:
        q = q.unsqueeze(0)

    quad_term = 0.5 * torch.bmm(l.unsqueeze(1), torch.bmm(H, l.unsqueeze(-1))).squeeze()
    lin_term = torch.sum(q * l, dim=-1)
    return quad_term + lin_term

def run_accuracy_check():
    torch.manual_seed(42)
    device = torch.device('cpu')
    
    # Problem dimensions
    k = 20
    dim_q = 3 * k
    eps = 1e-3
    n_batch = 1
    
    print(f"Comparing solvers on a single random instance (k={k}, eps={eps})")
    
    # Generate random problem data
    J = torch.randn(n_batch, dim_q, dim_q, device=device)
    q = torch.randn(n_batch, dim_q, device=device)
    
    # 1. PGD Solver (PyTorch)
    l_pgd = accelerated_pgd_socp_sappy(J, q, eps, max_iter=5000, tol=1e-9)
    obj_pgd = calculate_objective(l_pgd, J, q, eps)
    
    # 2. SAP Solver
    solver_sappy = SAPSolver()
    l_sap = solver_sappy.apply(J, q, eps)
    obj_sap = calculate_objective(l_sap, J, q, eps)
    
    # 3. JAX Solver
    J_jax = jnp.array(J.numpy())
    q_jax = jnp.array(q.numpy())
    l_jax_raw = accelerated_pgd_socp_sappy_jax(J_jax, q_jax, eps, max_iter=5000)
    l_jax = torch.from_numpy(jax.device_get(l_jax_raw)).float()
    obj_jax = calculate_objective(l_jax, J, q, eps)
    
    print("\n--- Results ---")
    print(f"PGD Objective: {obj_pgd.item():.8f}")
    print(f"SAP Objective: {obj_sap.item():.8f}")
    print(f"JAX Objective: {obj_jax.item():.8f}")
    
    print("\n--- Relative Objective Differences ---")
    print(f"|PGD - SAP|: {torch.abs(obj_pgd - obj_sap).item():.2e}")
    print(f"|PGD - JAX|: {torch.abs(obj_pgd - obj_jax).item():.2e}")
    print(f"|SAP - JAX|: {torch.abs(obj_sap - obj_jax).item():.2e}")
    
    print("\n--- Solution L2 Distance (Primal) ---")
    print(f"dist(PGD, SAP): {torch.norm(l_pgd - l_sap).item():.2e}")
    print(f"dist(PGD, JAX): {torch.norm(l_pgd - l_jax).item():.2e}")
    print(f"dist(SAP, JAX): {torch.norm(l_sap - l_jax).item():.2e}")

if __name__ == "__main__":
    run_accuracy_check()
