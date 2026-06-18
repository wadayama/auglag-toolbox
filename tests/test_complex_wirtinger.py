"""E5: complex (Wirtinger) variant of the covariance-weighted constraint.

    maximise  logdet(I + (H F)(H F)^H / sigma^2)        (complex F, H)
    s.t.      Re tr(F W F^H) <= P1                       (real-valued constraint)
              ||F||_F^2 <= P
"""

import torch
from pga_toolbox import pga_ascent_spg, project_frobenius_ball

from auglag_toolbox import augmented_lagrangian
from _helpers import logdet_pd, randc


def test_e5_complex_reaches_feasible_kkt():
    d, sigma, P = 3, 0.5, 4.0
    g = torch.Generator().manual_seed(11)
    H = torch.complex(
        torch.randn(d, d, generator=g, dtype=torch.float64),
        torch.randn(d, d, generator=g, dtype=torch.float64),
    )
    W = torch.diag(torch.tensor([2.5, 1.0, 0.4], dtype=torch.complex128))
    I = torch.eye(d, dtype=torch.complex128)

    def f_of(F):
        HF = H @ F
        return logdet_pd(I + HF @ HF.conj().T / sigma**2)

    def easy_proj(params):
        return [project_frobenius_ball(params[0], P)]

    F = randc(d, seed=12, scale=0.3).clone().requires_grad_(True)
    pga_ascent_spg(lambda: f_of(F), [F], projector=easy_proj,
                   max_iter=300, forward_budget=400)
    with torch.no_grad():
        P1 = 0.6 * float(torch.trace(F @ W @ F.conj().T).real)

    F = randc(d, seed=12, scale=0.3).clone().requires_grad_(True)

    def g1():
        return torch.trace(F @ W @ F.conj().T).real - P1

    res = augmented_lagrangian(
        lambda: f_of(F), [F], inequality=[g1], easy_projector=easy_proj,
    )

    assert res.feasibility <= 1e-6
    assert res.complementarity <= 1e-6
    assert res.stationarity <= 1e-5
    assert res.multipliers_ineq[0] > 1e-6
