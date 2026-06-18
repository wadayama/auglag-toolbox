"""AugLagResult.converged / stop_reason and user-tunable tolerances."""

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


def test_converged_flag_and_reason_on_success():
    d, W, f_of, easy_proj, P1 = _build_e1()
    F = randmat(d, seed=2, scale=0.3).clone().requires_grad_(True)
    g1 = lambda: torch.trace(F @ W @ F.T) - P1
    res = augmented_lagrangian(lambda: f_of(F), [F], inequality=[g1],
                               easy_projector=easy_proj)
    assert res.converged is True
    assert res.stop_reason == "kkt_tol"
    # All three KKT residuals are within the (default) tolerances.
    assert res.feasibility <= 1e-6
    assert res.complementarity <= 1e-6
    assert res.stationarity <= 1e-5


def test_not_converged_when_budget_exhausted():
    d, W, f_of, easy_proj, P1 = _build_e1()
    F = randmat(d, seed=2, scale=0.3).clone().requires_grad_(True)
    g1 = lambda: torch.trace(F @ W @ F.T) - P1
    # One outer iteration cannot satisfy the t >= 1 stopping guard.
    res = augmented_lagrangian(lambda: f_of(F), [F], inequality=[g1],
                               easy_projector=easy_proj, outer_iters=1)
    assert res.converged is False
    assert res.stop_reason == "max_outer_iters"
    assert res.outer_iters == 1


def test_tolerances_are_user_tunable():
    d, W, f_of, easy_proj, P1 = _build_e1()
    F = randmat(d, seed=2, scale=0.3).clone().requires_grad_(True)
    g1 = lambda: torch.trace(F @ W @ F.T) - P1
    # Tight stationarity tolerance is harder to hit within a small budget...
    tight = augmented_lagrangian(lambda: f_of(F), [F], inequality=[g1],
                                 easy_projector=easy_proj,
                                 stat_tol=1e-12, outer_iters=3)
    assert tight.converged is False
    # ...while a loose set of tolerances converges immediately.
    F = randmat(d, seed=2, scale=0.3).clone().requires_grad_(True)
    g1 = lambda: torch.trace(F @ W @ F.T) - P1
    loose = augmented_lagrangian(lambda: f_of(F), [F], inequality=[g1],
                                 easy_projector=easy_proj,
                                 feas_tol=1e-2, compl_tol=1e-2, stat_tol=1e-1)
    assert loose.converged is True


def test_penalty_reports_single_solve():
    d, W, f_of, easy_proj, P1 = _build_e1()
    F = randmat(d, seed=2, scale=0.3).clone().requires_grad_(True)
    g1 = lambda: torch.trace(F @ W @ F.T) - P1
    res = penalty_method(lambda: f_of(F), [F], inequality=[g1],
                         easy_projector=easy_proj, mu=1000.0)
    assert res.stop_reason == "single_penalty_solve"
    assert isinstance(res.converged, bool)
