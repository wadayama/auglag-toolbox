"""KKT residual diagnostics for constrained maximisation.

Productizes the reference ``kkt_metrics`` (smoke E1-E5), extended to handle
equality constraints. The three returned quantities are the standard
first-order optimality residuals for

    maximise f(x)   s.t.   g_c(x) <= 0,  h_k(x) = 0,  x in S_easy.
"""

from __future__ import annotations

import torch

from .types import Projector, ScalarClosure


def kkt_residuals(
    objective: ScalarClosure,
    params: list[torch.Tensor],
    inequality: list[ScalarClosure] | None,
    equality: list[ScalarClosure] | None,
    multipliers_ineq: list[float],
    multipliers_eq: list[float],
    easy_projector: Projector | None = None,
) -> tuple[float, float, float]:
    """Return ``(feasibility, complementarity, stationarity)`` at ``params``.

    - feasibility   = ``max( max_c [g_c]_+ , max_k |h_k| )``
    - complementarity = ``max_c |lambda_c * g_c|``  (equalities do not contribute)
    - stationarity  = projected-gradient-map residual of the Lagrangian
          ``L = f - sum_c lambda_c g_c - sum_k mu_k h_k``
      w.r.t. the easy feasible set:
          ``r = sum_p || p - P_easy(p + grad_p L) ||``.

    For complex parameters PyTorch's ``.grad`` is the Wirtinger gradient, so the
    residual is computed identically in the real and complex cases.
    """
    inequality = inequality or []
    equality = equality or []

    for p in params:
        if p.grad is not None:
            p.grad.zero_()

    L = objective()
    for c, g in enumerate(inequality):
        L = L - multipliers_ineq[c] * g()
    for k, h in enumerate(equality):
        L = L - multipliers_eq[k] * h()
    L.backward()

    with torch.no_grad():
        gvals = [float(g().detach().real) for g in inequality]
        hvals = [float(h().detach().real) for h in equality]
        feas_ineq = max((max(0.0, gv) for gv in gvals), default=0.0)
        feas_eq = max((abs(hv) for hv in hvals), default=0.0)
        feas = max(feas_ineq, feas_eq)
        compl = max(
            (abs(multipliers_ineq[c] * gvals[c]) for c in range(len(gvals))),
            default=0.0,
        )
        # Projected-gradient map residual (unit step).
        trial = [p + p.grad for p in params]
        if easy_projector is not None:
            out = easy_projector([t.clone() for t in trial])
            proj = out if out is not None else trial
        else:
            proj = trial
        stat = sum(float(torch.linalg.norm(p - q)) for p, q in zip(params, proj))

    return feas, compl, stat


def kkt_residuals_batched(
    objective: ScalarClosure,
    params: list[torch.Tensor],
    inequality: list[ScalarClosure] | None,
    equality: list[ScalarClosure] | None,
    multipliers_ineq: list[torch.Tensor],
    multipliers_eq: list[torch.Tensor],
    easy_projector: Projector | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-element ``(B,)`` KKT residuals for a batched (multi-start) solve.

    Same definitions as :func:`kkt_residuals`, vectorised over a leading batch
    dimension. ``objective`` / each constraint return ``(B,)``; multipliers are
    ``(B,)`` tensors; ``params`` are ``(B, *shape)``. Uses the single-backward
    trick ``L.sum().backward()`` (valid because batch elements are independent).
    Returns ``(feasibility, complementarity, stationarity)`` each shaped ``(B,)``.
    """
    inequality = inequality or []
    equality = equality or []
    B = params[0].shape[0]
    dev = params[0].device
    rdt = params[0].real.dtype if params[0].is_complex() else params[0].dtype

    for p in params:
        if p.grad is not None:
            p.grad.zero_()

    L = objective()
    for c, g in enumerate(inequality):
        L = L - multipliers_ineq[c] * g()
    for k, h in enumerate(equality):
        L = L - multipliers_eq[k] * h()
    L.sum().backward()

    with torch.no_grad():
        gvals = [g().detach().real for g in inequality]
        hvals = [h().detach().real for h in equality]
        terms = [torch.clamp(gv, min=0.0) for gv in gvals] + [hv.abs() for hv in hvals]
        feas = (torch.stack(terms, 0).amax(0) if terms
                else torch.zeros(B, dtype=rdt, device=dev))
        compl = (torch.stack([(multipliers_ineq[c] * gvals[c]).abs()
                              for c in range(len(gvals))], 0).amax(0)
                 if gvals else torch.zeros(B, dtype=rdt, device=dev))
        trial = [p + p.grad for p in params]
        if easy_projector is not None:
            out = easy_projector([t.clone() for t in trial])
            proj = out if out is not None else trial
        else:
            proj = trial
        stat = torch.zeros(B, dtype=rdt, device=dev)
        for p, q in zip(params, proj):
            stat = stat + ((p - q).abs() ** 2).flatten(1).sum(1).sqrt()

    return feas, compl, stat
