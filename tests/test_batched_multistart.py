"""Batched (parallel multi-start) ALM: correctness vs the sequential oracle,
multimodal value, and the result/winner plumbing.
"""

import torch
from pga_toolbox import (
    pga_ascent_spg,
    project_frobenius_ball,
    project_frobenius_ball_batched,
)

from auglag_toolbox import (
    AugLagResult,
    augmented_lagrangian,
    augmented_lagrangian_batched,
    augmented_lagrangian_descent_batched,
    kkt_residuals_batched,
)
from _helpers import logdet_pd, randmat


def _btrace(A):
    return torch.diagonal(A, dim1=-2, dim2=-1).sum(-1)


def test_batched_matches_sequential_oracle_e1():
    """Each batch element must match the verified library solve from same init."""
    d, sigma, P, B = 3, 0.5, 4.0, 6
    H = randmat(d, seed=1)
    W = torch.diag(torch.tensor([2.5, 1.0, 0.4], dtype=torch.float64))
    I = torch.eye(d, dtype=torch.float64)
    inits = [randmat(d, seed=100 + b, scale=0.3) for b in range(B)]

    # Active cap from an easy-only solve.
    F0 = inits[0].clone().requires_grad_(True)
    pga_ascent_spg(lambda: logdet_pd(I + (H @ F0) @ (H @ F0).T / sigma**2),
                   [F0], projector=lambda ps: [project_frobenius_ball(ps[0], P)],
                   max_iter=300, forward_budget=400)
    with torch.no_grad():
        P1 = 0.6 * float(_btrace(F0 @ W @ F0.T))

    # Sequential oracle: library ALM from each init.
    seq = []
    for b in range(B):
        F = inits[b].clone().requires_grad_(True)
        g1 = lambda F=F: _btrace(F @ W @ F.T) - P1
        r = augmented_lagrangian(
            lambda F=F: logdet_pd(I + (H @ F) @ (H @ F).T / sigma**2),
            [F], inequality=[g1],
            easy_projector=lambda ps: [project_frobenius_ball(ps[0], P)])
        seq.append(r)

    # Batched ALM on all B at once.
    Fb = torch.stack(inits, 0).clone().requires_grad_(True)
    obj_b = lambda: logdet_pd(I + torch.matmul(H, Fb) @ torch.matmul(H, Fb).transpose(-1, -2) / sigma**2)
    g1_b = lambda: _btrace(torch.matmul(torch.matmul(Fb, W), Fb.transpose(-1, -2))) - P1
    proj_b = lambda ps: [project_frobenius_ball_batched(ps[0], P)]

    res = augmented_lagrangian_batched(obj_b, [Fb], inequality=[g1_b],
                                       easy_projector=proj_b)

    assert bool(res.converged.all())
    assert float(res.feasibility.amax()) <= 1e-6
    for b in range(B):
        rel = abs(float(res.objective[b]) - seq[b].objective) / max(abs(seq[b].objective), 1e-12)
        assert rel < 1e-3
        assert abs(float(res.multipliers_ineq[0][b]) - seq[b].multipliers_ineq[0]) < 1e-3


def test_winner_result_is_global_incumbent():
    """On a multimodal problem, best-of-B reaches the library's global, not all do."""
    n, tilt, P, cap, B = 6, 0.5, 12.0, 5.5, 32

    # Reference global via the library's sequential multistart.
    x0 = (0.3 * torch.randn(n, generator=torch.Generator().manual_seed(0),
                            dtype=torch.float64)).requires_grad_(True)
    ref = augmented_lagrangian(
        lambda: (-(x0**2 - 1) ** 2 + tilt * x0).sum(), [x0],
        inequality=[lambda: (x0**2).sum() - cap],
        easy_projector=lambda ps: [project_frobenius_ball(ps[0], P)],
        multistart=80, seed=0)

    Xb = (0.6 * torch.randn(B, n, generator=torch.Generator().manual_seed(123),
                            dtype=torch.float64)).clone().requires_grad_(True)
    res = augmented_lagrangian_batched(
        lambda: (-(Xb**2 - 1) ** 2 + tilt * Xb).sum(-1), [Xb],
        inequality=[lambda: (Xb**2).sum(-1) - cap],
        easy_projector=lambda ps: [project_frobenius_ball_batched(ps[0], P)])

    assert float(res.feasibility.amax()) <= 1e-6
    # best-of-B reaches the reference global...
    assert abs(float(res.objective.amax()) - ref.objective) < 1e-3
    # ...but not every start does (genuine multimodality + independence).
    near = int((res.objective >= ref.objective - 1e-3).sum())
    assert near < B
    assert float(res.objective.amax() - res.objective.amin()) > 1e-2

    # winner_result extracts the incumbent as a scalar AugLagResult.
    wr = res.winner_result()
    assert isinstance(wr, AugLagResult)
    assert abs(wr.objective - float(res.objective[res.winner])) < 1e-12
    assert wr.converged is True


def test_descent_batched_minimises_per_element():
    d, sigma, P, B = 3, 0.5, 4.0, 4
    H = randmat(d, seed=1)
    W = torch.diag(torch.tensor([2.5, 1.0, 0.4], dtype=torch.float64))
    I = torch.eye(d, dtype=torch.float64)
    inits = [randmat(d, seed=100 + b, scale=0.3) for b in range(B)]

    F0 = inits[0].clone().requires_grad_(True)
    pga_ascent_spg(lambda: logdet_pd(I + (H @ F0) @ (H @ F0).T / sigma**2),
                   [F0], projector=lambda ps: [project_frobenius_ball(ps[0], P)],
                   max_iter=300, forward_budget=400)
    with torch.no_grad():
        P1 = 0.6 * float(_btrace(F0 @ W @ F0.T))

    Fb = torch.stack(inits, 0).clone().requires_grad_(True)
    cost_b = lambda: -logdet_pd(I + torch.matmul(H, Fb) @ torch.matmul(H, Fb).transpose(-1, -2) / sigma**2)
    g1_b = lambda: _btrace(torch.matmul(torch.matmul(Fb, W), Fb.transpose(-1, -2))) - P1
    proj_b = lambda ps: [project_frobenius_ball_batched(ps[0], P)]

    res = augmented_lagrangian_descent_batched(cost_b, [Fb], inequality=[g1_b],
                                               easy_projector=proj_b)
    assert float(res.feasibility.amax()) <= 1e-6
    # Reported objective is the original cost (negative log-det here).
    with torch.no_grad():
        assert torch.allclose(res.objective, cost_b().detach().real, atol=1e-6)


def test_kkt_residuals_batched_small_at_solution():
    d, sigma, P, B = 3, 0.5, 4.0, 4
    H = randmat(d, seed=1)
    W = torch.diag(torch.tensor([2.5, 1.0, 0.4], dtype=torch.float64))
    I = torch.eye(d, dtype=torch.float64)
    inits = [randmat(d, seed=100 + b, scale=0.3) for b in range(B)]
    F0 = inits[0].clone().requires_grad_(True)
    pga_ascent_spg(lambda: logdet_pd(I + (H @ F0) @ (H @ F0).T / sigma**2),
                   [F0], projector=lambda ps: [project_frobenius_ball(ps[0], P)],
                   max_iter=300, forward_budget=400)
    with torch.no_grad():
        P1 = 0.6 * float(_btrace(F0 @ W @ F0.T))

    Fb = torch.stack(inits, 0).clone().requires_grad_(True)
    obj_b = lambda: logdet_pd(I + torch.matmul(H, Fb) @ torch.matmul(H, Fb).transpose(-1, -2) / sigma**2)
    g1_b = lambda: _btrace(torch.matmul(torch.matmul(Fb, W), Fb.transpose(-1, -2))) - P1
    proj_b = lambda ps: [project_frobenius_ball_batched(ps[0], P)]
    res = augmented_lagrangian_batched(obj_b, [Fb], inequality=[g1_b], easy_projector=proj_b)

    feas, compl, stat = kkt_residuals_batched(
        obj_b, [Fb], [g1_b], None, res.multipliers_ineq, res.multipliers_eq, proj_b)
    assert feas.shape == (B,)
    assert float(feas.amax()) <= 1e-6
    assert float(compl.amax()) <= 1e-6
    assert float(stat.amax()) <= 1e-5
