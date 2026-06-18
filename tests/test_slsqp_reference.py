"""Cross-validate the E1 ALM optimum against scipy SLSQP (skipped if no scipy)."""

import numpy as np
import pytest
import torch
from pga_toolbox import pga_ascent_spg, project_frobenius_ball

from auglag_toolbox import augmented_lagrangian
from _helpers import logdet_pd, randmat

scipy_optimize = pytest.importorskip("scipy.optimize")


def _slsqp_reference_e1(H, W, sigma, P, P1, d, n_starts=8):
    from scipy.optimize import minimize

    Hn, Wn = H.numpy(), W.numpy()

    def neg_f(x):
        F = x.reshape(d, d)
        HF = Hn @ F
        M = np.eye(d) + HF @ HF.T / sigma**2
        return -np.linalg.slogdet(M)[1]

    cons = [
        {"type": "ineq", "fun": lambda x: P - float(x @ x)},
        {"type": "ineq",
         "fun": lambda x: P1 - float((x.reshape(d, d) @ Wn * x.reshape(d, d)).sum())},
    ]
    best = None
    for s in range(n_starts):
        rng = np.random.RandomState(100 + s)
        x0 = 0.3 * rng.randn(d * d)
        r = minimize(neg_f, x0, method="SLSQP", constraints=cons,
                     options={"maxiter": 300, "ftol": 1e-10})
        if not r.success:
            continue
        feas_ball = (P - r.x @ r.x) >= -1e-6
        feas_pw = (P1 - (r.x.reshape(d, d) @ Wn * r.x.reshape(d, d)).sum()) >= -1e-6
        if feas_ball and feas_pw:
            val = -r.fun
            if best is None or val > best:
                best = val
    return best


def test_e1_matches_slsqp():
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
    res = augmented_lagrangian(lambda: f_of(F), [F], inequality=[g1],
                               easy_projector=easy_proj)

    f_slsqp = _slsqp_reference_e1(H, W, sigma, P, P1, d)
    assert f_slsqp is not None
    rel = abs(res.objective - f_slsqp) / max(abs(f_slsqp), 1e-12)
    assert rel < 1e-3
