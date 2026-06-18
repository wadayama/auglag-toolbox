"""The inner solver is pluggable: the Armijo driver also reaches a feasible KKT.

Confirms the inner-solver contract (maximise the closure, update params in
place) holds for a non-default solver.
"""

import torch
from pga_toolbox import pga_ascent_armijo, pga_ascent_spg, project_frobenius_ball

from auglag_toolbox import augmented_lagrangian
from _helpers import logdet_pd, randmat


def test_armijo_inner_reaches_feasible_kkt():
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
        P1 = 0.6 * float(torch.trace(F @ W @ F.T))

    F = randmat(d, seed=2, scale=0.3).clone().requires_grad_(True)
    g1 = lambda: torch.trace(F @ W @ F.T) - P1

    # Armijo needs a larger per-iteration forward budget than SPG to solve the
    # inner subproblem accurately enough for the outer loop to converge tightly.
    res = augmented_lagrangian(
        lambda: f_of(F), [F], inequality=[g1], easy_projector=easy_proj,
        inner=pga_ascent_armijo, inner_kwargs={"max_iter": 1000, "forward_budget": 1000},
    )

    assert res.feasibility <= 1e-6
    assert res.complementarity <= 1e-6
    assert res.stationarity <= 1e-5
