"""Batched (GPU-parallel) multi-start PHR augmented Lagrangian.

Runs ``B`` independent ALM solves in parallel over a leading batch dimension --
the non-convexity safeguard of multi-start, but with the starts solved at once
instead of sequentially. The inner subproblem is delegated to a batched
first-order solver (default ``pga_toolbox.pga_ascent_spg_batched``); the outer
multiplier loop carries per-element ``lambda``, ``mu``, ``rho``, feasibility and
convergence state as ``(B,)`` tensors.

Validated against the sequential :func:`augmented_lagrangian` element-by-element
(see local_notes/batched-multistart, GO verdict).

Contract (differs from the sequential API):
  - ``objective`` and each constraint return a real ``(B,)`` tensor and must be
    **NaN-safe** (a diverged element returns NaN, never raises -- a batched
    cholesky / logdet cannot be per-element try/excepted).
  - ``params`` are leaves shaped ``(B, *shape)`` with ``requires_grad=True``;
    the ``B`` starts are the multi-start initial points.
  - ``easy_projector`` must be **batch-aware** (e.g.
    ``project_frobenius_ball_batched``), projecting each element independently.
"""

from __future__ import annotations

import torch
from pga_toolbox import pga_ascent_spg_batched

from .kkt import kkt_residuals_batched
from .types import BatchedAugLagResult, InnerSolver, Projector, ScalarClosure

_DEFAULT_INNER_KWARGS = {"max_iter": 300, "forward_budget": 400}


def augmented_lagrangian_batched(
    objective: ScalarClosure,
    params: list[torch.Tensor],
    *,
    inequality: list[ScalarClosure] | None = None,
    equality: list[ScalarClosure] | None = None,
    easy_projector: Projector | None = None,
    inner: InnerSolver = pga_ascent_spg_batched,
    inner_kwargs: dict | None = None,
    rho0: float = 1.0,
    rho_grow: float = 3.0,
    rho_max: float = 1e6,
    feas_tol: float = 1e-6,
    compl_tol: float = 1e-6,
    stat_tol: float = 1e-5,
    outer_iters: int = 40,
    verbose: bool = False,
) -> BatchedAugLagResult:
    """Maximise ``objective`` from ``B`` parallel starts subject to constraints.

    See the module docstring for the (batched) closure / params / projector
    contract. Returns a :class:`BatchedAugLagResult`; use ``.winner_result()``
    to get the global incumbent as a scalar ``AugLagResult``.

    Each start runs its own PHR outer loop with per-element multipliers and
    penalty. An element is marked converged once all three KKT residuals meet
    ``feas_tol`` / ``compl_tol`` / ``stat_tol``; converged elements have their
    dual updates frozen (retired) and the loop stops when all elements converge
    or ``outer_iters`` is exhausted.
    """
    inequality = list(inequality) if inequality else []
    equality = list(equality) if equality else []
    kw = dict(inner_kwargs) if inner_kwargs else dict(_DEFAULT_INNER_KWARGS)

    B = params[0].shape[0]
    dev = params[0].device
    rdt = params[0].real.dtype if params[0].is_complex() else params[0].dtype

    lam = [torch.zeros(B, dtype=rdt, device=dev) for _ in inequality]
    mu = [torch.zeros(B, dtype=rdt, device=dev) for _ in equality]
    rho = torch.full((B,), float(rho0), dtype=rdt, device=dev)
    prev_feas = torch.full((B,), float("inf"), dtype=rdt, device=dev)
    converged = torch.zeros(B, dtype=torch.bool, device=dev)
    total_inner_evals = 0
    history: list[dict] = []
    feas = compl = stat = torch.full((B,), float("inf"), dtype=rdt, device=dev)
    t = 0

    for t in range(outer_iters):
        counter = {"n": 0}

        def augmented(_lam=lam, _mu=mu, _rho=rho):
            counter["n"] += 1
            val = objective()                                    # (B,)
            for c, g in enumerate(inequality):
                viol = torch.clamp(_lam[c] + _rho * g(), min=0.0)
                val = val - (viol**2 - _lam[c] ** 2) / (2.0 * _rho)
            for k, h in enumerate(equality):
                hk = h()
                val = val - (_mu[k] * hk + (_rho / 2.0) * hk**2)
            return val                                           # (B,)

        inner(augmented, params, projector=easy_projector, **kw)
        total_inner_evals += counter["n"]

        # Dual update, frozen for already-converged (retired) elements.
        with torch.no_grad():
            gv = [g().detach().real for g in inequality]
            hv = [h().detach().real for h in equality]
        active = ~converged
        for c in range(len(lam)):
            updated = torch.clamp(lam[c] + rho * gv[c], min=0.0)
            lam[c] = torch.where(active, updated, lam[c])
        for k in range(len(mu)):
            updated = mu[k] + rho * hv[k]
            mu[k] = torch.where(active, updated, mu[k])

        feas, compl, stat = kkt_residuals_batched(
            objective, params, inequality, equality, lam, mu, easy_projector
        )
        if t >= 1:
            newly = (feas <= feas_tol) & (compl <= compl_tol) & (stat <= stat_tol)
            converged = converged | newly

        history.append({
            "feas": feas.detach().clone(), "compl": compl.detach().clone(),
            "stat": stat.detach().clone(), "rho": rho.detach().clone(),
            "converged": converged.clone(), "inner_evals": counter["n"],
        })
        if verbose:
            print(f"  [outer {t:2d}] converged={int(converged.sum())}/{B} "
                  f"max_feas={float(feas.amax()):.3e} "
                  f"max_stat={float(stat.amax()):.3e}")

        if bool(converged.all()):
            break
        stalled = (feas > 0.5 * prev_feas) & active
        rho = torch.where(stalled, torch.clamp(rho * rho_grow, max=rho_max), rho)
        prev_feas = feas

    with torch.no_grad():
        obj = objective().detach().real

    feasible = feas <= feas_tol
    if bool(feasible.any()):
        score = torch.where(feasible, obj, torch.full_like(obj, float("-inf")))
        winner = int(score.argmax())
    else:
        winner = int(feas.argmin())

    return BatchedAugLagResult(
        params=params,
        objective=obj,
        multipliers_ineq=lam,
        multipliers_eq=mu,
        feasibility=feas,
        complementarity=compl,
        stationarity=stat,
        rho=rho,
        converged=converged,
        winner=winner,
        outer_iters=t + 1,
        inner_evals_total=total_inner_evals,
        history=history,
    )


def augmented_lagrangian_descent_batched(
    cost: ScalarClosure,
    params: list[torch.Tensor],
    **kwargs,
) -> BatchedAugLagResult:
    """Batched minimisation of ``cost`` (returning ``(B,)``) via ``max(-cost)``.

    Accepts the same keyword arguments as :func:`augmented_lagrangian_batched`.
    The returned ``objective`` tensor is the original per-element cost.
    """
    result = augmented_lagrangian_batched(lambda: -cost(), params, **kwargs)
    result.objective = -result.objective
    return result
