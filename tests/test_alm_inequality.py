"""E1: single covariance-weighted inequality constraint (real), active KKT point.

    maximise  logdet(I + (H F)(H F)^T / sigma^2)
    s.t.      tr(F W F^T) <= P1   (W != I, no closed-form projection)
              ||F||_F^2 <= P      (easy: Frobenius ball)
"""

import torch
from pga_toolbox import pga_ascent_spg, project_frobenius_ball

from auglag_toolbox import augmented_lagrangian
from _helpers import logdet_pd, randmat


def _build_e1():
    d, sigma, P = 3, 0.5, 4.0
    H = randmat(d, seed=1)
    W = torch.diag(torch.tensor([2.5, 1.0, 0.4], dtype=torch.float64))

    def f_of(F):
        HF = H @ F
        return logdet_pd(torch.eye(d, dtype=torch.float64) + HF @ HF.T / sigma**2)

    def easy_proj(params):
        return [project_frobenius_ball(params[0], P)]

    # Auto-pick P1 from the easy-only optimum so the hard constraint is ACTIVE.
    F = randmat(d, seed=2, scale=0.3).clone().requires_grad_(True)
    pga_ascent_spg(lambda: f_of(F), [F], projector=easy_proj,
                   max_iter=300, forward_budget=400)
    with torch.no_grad():
        P1 = 0.6 * float(torch.trace(F @ W @ F.T))
    return d, sigma, P, H, W, f_of, easy_proj, P1


def test_e1_reaches_feasible_active_kkt():
    d, sigma, P, H, W, f_of, easy_proj, P1 = _build_e1()
    F = randmat(d, seed=2, scale=0.3).clone().requires_grad_(True)

    def g1():
        return torch.trace(F @ W @ F.T) - P1

    res = augmented_lagrangian(
        lambda: f_of(F), [F], inequality=[g1], easy_projector=easy_proj,
    )

    assert res.feasibility <= 1e-6
    assert res.complementarity <= 1e-6
    assert res.stationarity <= 1e-5
    # Constraint must be active: positive multiplier and binding power.
    assert res.multipliers_ineq[0] > 1e-6
    with torch.no_grad():
        power = float(torch.trace(F @ W @ F.T))
    assert abs(power - P1) / P1 < 1e-3
