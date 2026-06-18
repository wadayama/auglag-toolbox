"""Visual abstract for auglag-toolbox (two panels, both from actual runs).

  A  Augmented Lagrangian enforces a projection-free constraint: on the coupled,
     covariance-dependent relay node-power problem (E2), the KKT residuals
     (feasibility / complementarity / stationarity) collapse to ~1e-7 at a
     finite rho, i.e. the solver reaches a feasible KKT point.
  B  Parallel multi-start escapes local optima: on a multimodal constrained
     problem, best-of-B (the winner) reaches the global optimum while many of
     the B independent starts get stuck at worse local optima.

Nothing is hard-coded -- every plotted number comes from the optimiser.

Run:
    uv run --extra examples python examples/visual_abstract.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch

from pga_toolbox import (
    pga_ascent_spg,
    project_frobenius_ball,
    project_frobenius_ball_batched,
)

from auglag_toolbox import augmented_lagrangian, augmented_lagrangian_batched

torch.set_default_dtype(torch.float64)
FIG = pathlib.Path(__file__).resolve().parent.parent / "docs" / "figures"


def logdet_pd(A):
    A = 0.5 * (A + A.transpose(-1, -2).conj())
    return torch.linalg.slogdet(A).logabsdet


def randmat(d, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return scale * torch.randn(d, d, generator=g, dtype=torch.float64)


def btrace(A):
    return torch.diagonal(A, dim1=-2, dim2=-1).sum(-1)


def panel_a_data():
    """Coupled, covariance-dependent relay node-power constraint (E2)."""
    d, sigma, Pf = 2, 0.5, 4.0
    H2 = randmat(d, seed=3)
    I = torch.eye(d, dtype=torch.float64)

    def mi(F, R):
        G = H2 @ R @ F
        C = sigma**2 * (H2 @ R @ R.T @ H2.T) + sigma**2 * I
        return logdet_pd(G @ G.T + C) - logdet_pd(C)

    def easy_proj(params):
        F, R = params
        return [project_frobenius_ball(F, Pf), R]

    F = randmat(d, seed=4, scale=0.3).clone().requires_grad_(True)
    R = randmat(d, seed=5, scale=0.3).clone().requires_grad_(True)
    pga_ascent_spg(lambda: mi(F, R), [F, R], projector=easy_proj,
                   max_iter=400, forward_budget=600)
    with torch.no_grad():
        P_relay = 2.0 * float(torch.trace(F @ F.T + sigma**2 * I))

    F = randmat(d, seed=4, scale=0.3).clone().requires_grad_(True)
    R = randmat(d, seed=5, scale=0.3).clone().requires_grad_(True)

    def g_relay():
        K11 = F @ F.T + sigma**2 * I
        return torch.trace(R @ K11 @ R.T) - P_relay

    res = augmented_lagrangian(lambda: mi(F, R), [F, R],
                               inequality=[g_relay], easy_projector=easy_proj)
    feas = np.array([h["feas"] for h in res.history])
    compl = np.array([h["compl"] for h in res.history])
    stat = np.array([h["stat"] for h in res.history])
    return feas, compl, stat, res.objective, res.multipliers_ineq[0], res.rho


def panel_b_data():
    """Multimodal tilted double well with an active quadratic constraint."""
    n, tilt, P, cap, B = 6, 0.5, 12.0, 5.5, 64
    Xb = (0.6 * torch.randn(B, n, generator=torch.Generator().manual_seed(123),
                            dtype=torch.float64)).clone().requires_grad_(True)
    res = augmented_lagrangian_batched(
        lambda: (-(Xb**2 - 1) ** 2 + tilt * Xb).sum(-1), [Xb],
        inequality=[lambda: (Xb**2).sum(-1) - cap],
        easy_projector=lambda ps: [project_frobenius_ball_batched(ps[0], P)])
    obj = res.objective.detach().numpy()
    return obj, res.winner, B


def main():
    feas, compl, stat, mi_star, lam, rho = panel_a_data()
    obj, winner, B = panel_b_data()
    glob = float(obj.max())
    near = int((obj >= glob - 1e-3).sum())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 3.8))

    # ---- Panel A: KKT convergence ----
    it = np.arange(1, len(feas) + 1)
    axA.semilogy(it, np.maximum(feas, 1e-16), "o-", color="#4477aa", label="feasibility")
    axA.semilogy(it, np.maximum(compl, 1e-16), "s-", color="#ee6677", label="complementarity")
    axA.semilogy(it, np.maximum(stat, 1e-16), "^-", color="#228833", label="stationarity")
    axA.axhline(1e-6, color="gray", ls=":", lw=1)
    axA.set_xlabel("outer iteration")
    axA.set_ylabel("KKT residual")
    axA.set_title("A  Projection-free constraint $\\to$ KKT point", fontsize=11, loc="left")
    axA.set_xticks(it)
    axA.legend(fontsize=8, loc="upper right")
    axA.grid(True, which="both", alpha=0.25)
    axA.text(0.04, 0.06,
             f"coupled $\\mathrm{{tr}}(R K_{{11}} R^H)\\leq P$, active\n"
             f"MI*={mi_star:.3f}, $\\lambda$={lam:.3f}, finite $\\rho$={rho:.0f}",
             transform=axA.transAxes, fontsize=8, va="bottom",
             bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))

    # ---- Panel B: multistart escapes local optima ----
    order = np.argsort(obj)
    rank = np.arange(B)
    is_win = order == winner
    axB.scatter(rank[~is_win], obj[order][~is_win], s=18, color="#aab2bd",
                label="single start", zorder=2)
    axB.axhline(glob, color="#ee6677", ls="--", lw=1.2,
                label=f"global (best-of-{B})")
    wpos = int(np.where(order == winner)[0][0])
    axB.scatter([wpos], [obj[winner]], s=150, marker="*", color="#ee6677",
                edgecolor="k", linewidth=0.5, zorder=3, label="winner")
    axB.set_xlabel("start (sorted by objective)")
    axB.set_ylabel("objective at convergence")
    axB.set_title("B  Parallel multi-start escapes local optima", fontsize=11, loc="left")
    axB.legend(fontsize=8, loc="upper left")
    axB.grid(True, alpha=0.25)
    axB.text(0.96, 0.06,
             f"{near}/{B} starts reach global\nspread = {glob - float(obj.min()):.2f}",
             transform=axB.transAxes, fontsize=8, va="bottom", ha="right",
             bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "visual_abstract.png"
    fig.savefig(out, dpi=150)
    print(f"panel A: feas {feas[-1]:.1e} compl {compl[-1]:.1e} stat {stat[-1]:.1e} "
          f"MI*={mi_star:.4f} lambda={lam:.4f} rho={rho:.0f}")
    print(f"panel B: global={glob:.4f}  winner={winner}  reach-global {near}/{B}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
