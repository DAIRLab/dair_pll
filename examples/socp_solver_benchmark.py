import torch
import time
from dair_pll.socp_solver import accelerated_pgd_socp_sappy
from sappy import SAPSolver

def check_kkt(l, J, q, eps):
    """
    Verifies the KKT conditions for the SOCP problem:
    min 1/2 l^T H l + q^T l subject to l in K
    
    KKT conditions:
    1. Primal feasibility: l in K
    2. Dual feasibility: g = H l + q in K
    3. Complementary slackness: l^T g = 0
    """
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
        (100, 1000, 1e-3),
    ]
    
    solver_sappy = SAPSolver()
    
    for k, n_batch, eps in test_cases:
        dim_q = 3 * k
        # Random J of size (n_batch, dim_q, dim_q)
        J = torch.randn(n_batch, dim_q, dim_q, device=device)
        q = torch.randn(n_batch, dim_q, device=device)
        
        # Benchmark PGD (dair_pll implementation)
        start_time = time.time()
        l_pgd = accelerated_pgd_socp_sappy(J, q, eps, max_iter=1000, tol=1e-7)
        end_time = time.time()
        duration_pgd = end_time - start_time
        kkt_pgd = check_kkt(l_pgd, J, q, eps)
        
        # Benchmark SAPSolver (sappy implementation)
        # Note: SAPSolver.apply expect q to be (n_batch, dim_q) and J to be (n_batch, dim_q, dim_q)
        # It returns l of shape (n_batch, dim_q)
        start_time = time.time()
        l_sap = solver_sappy.apply(J, q, eps)
        end_time = time.time()
        duration_sap = end_time - start_time
        kkt_sap = check_kkt(l_sap, J, q, eps)
        
        print(f"\nCase: k={k}, batch={n_batch}, eps={eps}")
        print(f"  PGD Solver:")
        print(f"    Time: {duration_pgd:.4f}s ({n_batch / duration_pgd:.2f} samples/s)")
        print(f"    Primal Feas: {kkt_pgd['primal_feas_min']:.2e}, Dual Feas: {kkt_pgd['dual_feas_min']:.2e}, Comp: {kkt_pgd['complementarity']:.2e}")
        print(f"  SAP Solver:")
        print(f"    Time: {duration_sap:.4f}s ({n_batch / duration_sap:.2f} samples/s)")
        print(f"    Primal Feas: {kkt_sap['primal_feas_min']:.2e}, Dual Feas: {kkt_sap['dual_feas_min']:.2e}, Comp: {kkt_sap['complementarity']:.2e}")
        print(f"  Speedup (PGD/SAP): {duration_pgd / duration_sap:.2f}x")

if __name__ == "__main__":
    run_benchmark()
