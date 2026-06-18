"""E3: pure penalty (mu-sweep) vs ALM on the E1 problem.

Regression of the qualitative contrast:
  - small mu  -> infeasible;
  - large mu  -> feasible but the objective degrades (ill-conditioning);
  - ALM       -> feasible AND retains a high objective at finite rho.
"""

import torch
from pga_toolbox import pga_ascent_spg, project_frobenius_ball

from auglag_toolbox import augmented_lagrangian, penalty_method
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


def _g(F, W, P1):
    return lambda: torch.trace(F @ W @ F.T) - P1


def test_penalty_vs_alm_contrast():
    d, W, f_of, easy_proj, P1 = _build_e1()

    F1 = randmat(d, seed=2, scale=0.3).clone().requires_grad_(True)
    pen_small = penalty_method(lambda: f_of(F1), [F1], inequality=[_g(F1, W, P1)],
                               easy_projector=easy_proj, mu=1.0)

    F2 = randmat(d, seed=2, scale=0.3).clone().requires_grad_(True)
    pen_large = penalty_method(lambda: f_of(F2), [F2], inequality=[_g(F2, W, P1)],
                               easy_projector=easy_proj, mu=1000.0)

    Fa = randmat(d, seed=2, scale=0.3).clone().requires_grad_(True)
    alm = augmented_lagrangian(lambda: f_of(Fa), [Fa], inequality=[_g(Fa, W, P1)],
                               easy_projector=easy_proj)

    # Small mu leaves the constraint violated; large mu restores feasibility.
    assert pen_small.feasibility > 1e-3
    assert pen_large.feasibility < pen_small.feasibility

    # ALM is feasible AND keeps a higher objective than the large-mu penalty.
    assert alm.feasibility <= 1e-6
    assert alm.objective > pen_large.objective
    assert alm.rho <= 1e6
