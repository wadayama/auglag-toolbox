"""Equality-constraint ALM (new code, not covered by the smoke E1-E5).

Verifies the PHR equality term  -(mu h + (rho/2) h^2)  and the sign-free dual
update  mu <- mu + rho h  drive an equality constraint to satisfaction.

    maximise  logdet(I + (H F)(H F)^T / sigma^2)
    s.t.      tr(F F^T) = P_eq            (equality, multiplier mu free in sign)
"""

import torch
from pga_toolbox import project_frobenius_ball

from auglag_toolbox import augmented_lagrangian
from _helpers import logdet_pd, randmat


def test_equality_constraint_is_satisfied_at_kkt():
    d, sigma, P_eq = 3, 0.5, 2.0
    H = randmat(d, seed=1)

    def f_of(F):
        HF = H @ F
        return logdet_pd(torch.eye(d, dtype=torch.float64) + HF @ HF.T / sigma**2)

    F = randmat(d, seed=7, scale=0.3).clone().requires_grad_(True)

    def h1():
        return torch.trace(F @ F.T) - P_eq

    # A loose Frobenius ball that does not bind (keeps the inner solve stable);
    # the active constraint is the equality, handled by the multiplier.
    def easy_proj(params):
        return [project_frobenius_ball(params[0], 100.0)]

    res = augmented_lagrangian(
        lambda: f_of(F), [F], equality=[h1], easy_projector=easy_proj,
    )

    assert res.feasibility <= 1e-6
    assert res.stationarity <= 1e-5
    with torch.no_grad():
        power = float(torch.trace(F @ F.T))
    assert abs(power - P_eq) < 1e-6
    # Equality multiplier should be nonzero (constraint is active by construction).
    assert abs(res.multipliers_eq[0]) > 1e-6
