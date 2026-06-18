"""Sign conventions and KKT diagnostics.

  - g_c(x) <= 0, h_k(x) = 0 conventions;
  - inactive inequality  -> multiplier ~ 0, feasibility 0;
  - active inequality    -> multiplier > 0;
  - kkt_residuals returns small residuals at the ALM solution.
"""

import torch
from pga_toolbox import pga_ascent_spg, project_frobenius_ball

from auglag_toolbox import augmented_lagrangian, kkt_residuals
from _helpers import logdet_pd, randmat


def _build(P1_factor):
    d, sigma, P = 3, 0.5, 4.0
    H = randmat(d, seed=1)
    W = torch.diag(torch.tensor([2.5, 1.0, 0.4], dtype=torch.float64))

    def f_of(F):
        HF = H @ F
        return logdet_pd(torch.eye(d, dtype=torch.float64) + HF @ HF.T / sigma**2)

    def easy_proj(params):
        return [project_frobenius_ball(params[0], P)]

    F = randmat(d, seed=2, scale=0.3).clone().requires_grad_(True)
    pga_ascent_spg(lambda: f_of(F), [F], projector=easy_proj,
                   max_iter=300, forward_budget=400)
    with torch.no_grad():
        pw_easy = float(torch.trace(F @ W @ F.T))
    P1 = P1_factor * pw_easy
    return d, W, f_of, easy_proj, P1


def test_active_inequality_has_positive_multiplier():
    d, W, f_of, easy_proj, P1 = _build(0.6)  # tight -> active
    F = randmat(d, seed=2, scale=0.3).clone().requires_grad_(True)
    g1 = lambda: torch.trace(F @ W @ F.T) - P1
    res = augmented_lagrangian(lambda: f_of(F), [F], inequality=[g1],
                               easy_projector=easy_proj)
    assert res.feasibility <= 1e-6
    assert res.multipliers_ineq[0] > 1e-6


def test_inactive_inequality_drives_multiplier_to_zero():
    # P1 well above the easy-only power: the constraint never binds.
    d, W, f_of, easy_proj, P1 = _build(2.0)
    F = randmat(d, seed=2, scale=0.3).clone().requires_grad_(True)
    g1 = lambda: torch.trace(F @ W @ F.T) - P1
    res = augmented_lagrangian(lambda: f_of(F), [F], inequality=[g1],
                               easy_projector=easy_proj)
    assert res.feasibility <= 1e-6          # g1 < 0 (strictly satisfied)
    assert res.multipliers_ineq[0] < 1e-6   # inactive -> lambda ~ 0


def test_kkt_residuals_standalone_matches_solution():
    d, W, f_of, easy_proj, P1 = _build(0.6)
    F = randmat(d, seed=2, scale=0.3).clone().requires_grad_(True)
    g1 = lambda: torch.trace(F @ W @ F.T) - P1
    res = augmented_lagrangian(lambda: f_of(F), [F], inequality=[g1],
                               easy_projector=easy_proj)

    feas, compl, stat = kkt_residuals(
        lambda: f_of(F), [F], [g1], None,
        res.multipliers_ineq, res.multipliers_eq, easy_proj,
    )
    assert feas <= 1e-6
    assert compl <= 1e-6
    assert stat <= 1e-5
