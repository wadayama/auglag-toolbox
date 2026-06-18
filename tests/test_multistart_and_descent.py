"""E4-style multistart robustness and the descent wrapper.

  - sequential multistart returns a feasible point and is robust to rho schedule;
  - augmented_lagrangian_descent minimises a cost subject to the same constraints.
"""

import torch
from pga_toolbox import pga_ascent_spg, project_frobenius_ball

from auglag_toolbox import augmented_lagrangian, augmented_lagrangian_descent
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

    F = randmat(d, seed=2, scale=0.3).clone().requires_grad_(True)
    pga_ascent_spg(lambda: f_of(F), [F], projector=easy_proj,
                   max_iter=300, forward_budget=400)
    with torch.no_grad():
        P1 = 0.6 * float(torch.trace(F @ W @ F.T))
    return d, W, f_of, easy_proj, P1


def test_multistart_is_robust_to_rho_schedule():
    d, W, f_of, easy_proj, P1 = _build_e1()
    objectives = []
    for rho0, grow in ((0.3, 2.0), (1.0, 3.0), (3.0, 5.0)):
        F = randmat(d, seed=2, scale=0.3).clone().requires_grad_(True)
        g1 = lambda: torch.trace(F @ W @ F.T) - P1
        res = augmented_lagrangian(
            lambda: f_of(F), [F], inequality=[g1], easy_projector=easy_proj,
            rho0=rho0, rho_grow=grow, multistart=4, seed=2,
        )
        assert res.feasibility <= 1e-6
        objectives.append(res.objective)
    # All schedules find essentially the same feasible optimum.
    assert max(objectives) - min(objectives) < 1e-3


def test_descent_wrapper_minimises_cost():
    # Minimise cost = -f (so the optimum is the same E1 maximiser of f), s.t. g1<=0.
    d, W, f_of, easy_proj, P1 = _build_e1()
    F = randmat(d, seed=2, scale=0.3).clone().requires_grad_(True)
    g1 = lambda: torch.trace(F @ W @ F.T) - P1
    res = augmented_lagrangian_descent(
        lambda: -f_of(F), [F], inequality=[g1], easy_projector=easy_proj,
    )
    assert res.feasibility <= 1e-6
    assert res.stationarity <= 1e-5
    # Reported objective is the original cost = -f(x*) (a negative number here).
    with torch.no_grad():
        assert abs(res.objective - float(-f_of(F))) < 1e-6
