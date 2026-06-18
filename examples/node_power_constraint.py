"""Flagship example: node transmit-power constraint on a 2-hop relay.

Two-hop amplify-and-forward relay  X --H1--> [relay R] --H2--> Y.  We maximise
the end-to-end mutual information by jointly shaping the source precoder F and
the relay matrix R, subject to a *relay node transmit-power* budget

    tr( R K11 R^H ) <= P_relay,     K11 = F F^H + sigma^2 I,

which is nonlinear, coupled (depends on F through K11), covariance-dependent,
and has no closed-form projection. It is therefore handled by the augmented
Lagrangian (outer multiplier loop), while the source Frobenius-power budget
||F||_F^2 <= Pf is an easy constraint handled by projection inside the inner
solver. This is exactly the "future work" constraint sketched in the
gaussian-dag paper, demonstrated here end-to-end.

The covariance forward pass is written out explicitly so the example is
self-contained (no gaussian-dag dependency). All printed/plotted/saved numbers
come from the actual optimiser run -- nothing is hard-coded.

Run:
    uv run --extra examples python examples/node_power_constraint.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch

from pga_toolbox import pga_ascent_spg, project_frobenius_ball

from auglag_toolbox import augmented_lagrangian

torch.set_default_dtype(torch.float64)

HERE = pathlib.Path(__file__).resolve().parent


def logdet_pd(A):
    A = 0.5 * (A + A.transpose(-1, -2).conj())
    return torch.linalg.slogdet(A).logabsdet


def randmat(d, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return scale * torch.randn(d, d, generator=g, dtype=torch.float64)


def main():
    d, sigma, Pf = 2, 0.5, 4.0
    H2 = randmat(d, seed=3)
    I = torch.eye(d, dtype=torch.float64)

    def mi(F, R):
        # Effective channel G and effective noise covariance C of Y given X.
        G = H2 @ R @ F
        C = sigma**2 * (H2 @ R @ R.T @ H2.T) + sigma**2 * I
        return logdet_pd(G @ G.T + C) - logdet_pd(C)

    def easy_proj(params):  # project F onto the Frobenius ball; R is free
        F, R = params
        return [project_frobenius_ball(F, Pf), R]

    # --- Pick a binding cap P_relay near an R=I operating point -------------
    # MI saturates in the relay gain, so the unconstrained optimum drives the
    # relay power arbitrarily large. Capping near tr(K11) (x2) therefore binds.
    F = randmat(d, seed=4, scale=0.3).clone().requires_grad_(True)
    R = randmat(d, seed=5, scale=0.3).clone().requires_grad_(True)
    pga_ascent_spg(lambda: mi(F, R), [F, R], projector=easy_proj,
                   max_iter=400, forward_budget=600)
    with torch.no_grad():
        K11 = F @ F.T + sigma**2 * I
        P_relay = 2.0 * float(torch.trace(K11))
        mi_proj_only = float(mi(F, R))
        relay_power_proj_only = float(torch.trace(R @ K11 @ R.T))

    print(f"[B0 projection-only] MI={mi_proj_only:.4f}  "
          f"relay power={relay_power_proj_only:.4f}  (cap P_relay={P_relay:.4f})")

    # --- Augmented-Lagrangian solve -----------------------------------------
    F = randmat(d, seed=4, scale=0.3).clone().requires_grad_(True)
    R = randmat(d, seed=5, scale=0.3).clone().requires_grad_(True)

    def g_relay():
        K11 = F @ F.T + sigma**2 * I
        return torch.trace(R @ K11 @ R.T) - P_relay

    res = augmented_lagrangian(
        lambda: mi(F, R), [F, R], inequality=[g_relay], easy_projector=easy_proj,
        verbose=True,
    )

    with torch.no_grad():
        relay_power_alm = float(torch.trace(R @ (F @ F.T + sigma**2 * I) @ R.T))
    active = res.multipliers_ineq[0] > 1e-6 and \
        abs(relay_power_alm - P_relay) / P_relay < 1e-3

    print(f"\n[ALM] MI={res.objective:.5f}  feas={res.feasibility:.2e} "
          f"compl={res.complementarity:.2e} stat={res.stationarity:.2e}")
    print(f"      lambda={res.multipliers_ineq[0]:.4f}  rho={res.rho:.1e}  "
          f"outer_iters={res.outer_iters}  inner_evals={res.inner_evals_total}")
    print(f"      relay power={relay_power_alm:.4f} (cap {P_relay:.4f}) -> "
          f"constraint {'ACTIVE' if active else 'inactive'}")

    # --- Save results (real numbers only) -----------------------------------
    feas_hist = np.array([h["feas"] for h in res.history])
    stat_hist = np.array([h["stat"] for h in res.history])
    compl_hist = np.array([h["compl"] for h in res.history])
    rho_hist = np.array([h["rho"] for h in res.history])

    npz_path = HERE / "node_power_constraint_results.npz"
    np.savez(
        npz_path,
        d=d, sigma=sigma, Pf=Pf, P_relay=P_relay,
        mi_proj_only=mi_proj_only, relay_power_proj_only=relay_power_proj_only,
        mi_alm=res.objective, relay_power_alm=relay_power_alm,
        feasibility=res.feasibility, complementarity=res.complementarity,
        stationarity=res.stationarity, multiplier=res.multipliers_ineq[0],
        rho_final=res.rho, outer_iters=res.outer_iters,
        inner_evals_total=res.inner_evals_total,
        feas_hist=feas_hist, stat_hist=stat_hist,
        compl_hist=compl_hist, rho_hist=rho_hist,
    )
    print(f"\nsaved results -> {npz_path}")

    _plot(res, feas_hist, stat_hist, P_relay,
          relay_power_proj_only, relay_power_alm)


def _plot(res, feas_hist, stat_hist, P_relay, power_b0, power_alm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outer = np.arange(1, len(feas_hist) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))

    ax1.semilogy(outer, np.maximum(feas_hist, 1e-16), "o-", label="feasibility")
    ax1.semilogy(outer, np.maximum(stat_hist, 1e-16), "s-", label="stationarity")
    ax1.set_xlabel("outer iteration")
    ax1.set_ylabel("KKT residual")
    ax1.set_title("Convergence to a KKT point")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    ax2.bar([0, 1], [power_b0, power_alm],
            color=["#cc6677", "#4477aa"], width=0.6)
    ax2.axhline(P_relay, color="k", ls="--", label=f"cap $P_{{relay}}$={P_relay:.2f}")
    ax2.set_yscale("log")  # B0 runs away (~1e7) while ALM sits at the cap (9.0)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["projection-only\n(runs away)", "ALM\n(at cap)"])
    ax2.set_ylabel(r"relay power $\mathrm{tr}(R K_{11} R^H)$")
    ax2.set_title("Relay node power vs budget")
    ax2.legend()

    fig.tight_layout()
    pdf_path = HERE / "node_power_constraint.pdf"
    fig.savefig(pdf_path)
    print(f"saved figure  -> {pdf_path}")


if __name__ == "__main__":
    main()
