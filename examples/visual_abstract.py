"""Conceptual visual abstract for auglag-toolbox (a schematic, not a data plot).

  Left  : the two-layer architecture -- an outer augmented-Lagrangian loop
          (multipliers for projection-free constraints) wrapping an inner
          projected first-order solver (pga-toolbox) that handles the easy,
          projectable constraints.
  Right : the geometry -- the feasible region is the easy set S_easy (a ball,
          projectable) intersected with a hard, projection-free constraint
          g(x) <= 0. The inner solver projects onto S_easy; the outer multiplier
          pulls the iterate onto the active hard constraint, converging to a KKT
          point where grad f = lambda * grad g.

This is an illustration (no experimental numbers). Run to regenerate the figure:

    uv run --extra examples python examples/visual_abstract.py
"""

from __future__ import annotations

import pathlib

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon

FIG = pathlib.Path(__file__).resolve().parent.parent / "docs" / "figures"

C_OUTER = "#f5d9b0"   # auglag (outer)
C_INNER = "#bcd4ea"   # pga (inner)
C_EASY = "#cfe3f3"    # S_easy disk
C_INFEAS = "#f4c9c9"  # infeasible region
C_OBJ = "#9aa3ad"     # objective contours
C_ACC = "#c0392b"     # accents (KKT, constraint)


def _box(ax, x, y, w, h, fc, title, lines):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                fc=fc, ec="#5b6470", lw=1.4, zorder=2))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top",
            fontsize=11, fontweight="bold", zorder=3)
    ax.text(x + w / 2, y + h - 0.78, lines, ha="center", va="top",
            fontsize=8.4, zorder=3, color="#222")


def _arrow(ax, p0, p1, color="#5b6470", lw=2.0, style="-|>", ms=14):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                 lw=lw, color=color, zorder=4))


def architecture(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("A   Two-layer hybrid", fontsize=12, loc="left", fontweight="bold")

    ax.text(5, 9.5, r"maximize  $f(x)$   s.t.   $g(x)\leq 0,\ h(x)=0,\ x\in S_{\mathrm{easy}}$",
            ha="center", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round", fc="#fbfbfb", ec="0.8"))

    _box(ax, 1.0, 5.5, 8.0, 2.5, C_OUTER, "auglag-toolbox  ·  outer loop",
         "multipliers $\\lambda,\\mu$   ·   penalty $\\rho$\n"
         "KKT residuals (feas / compl / stat)   ·   converged")
    _box(ax, 1.0, 1.7, 8.0, 2.5, C_INNER, "pga-toolbox  ·  inner solver",
         "projected gradient  (SPG / Armijo)\n"
         "projection onto $S_{\\mathrm{easy}}$   ·   Wirtinger (complex)")

    _arrow(ax, (3.3, 5.5), (3.3, 4.2))           # outer -> inner
    ax.text(3.05, 4.85, "augmented $F_\\rho$\n+ projector", ha="right", va="center",
            fontsize=8.2, color="#333")
    _arrow(ax, (6.7, 4.2), (6.7, 5.5))           # inner -> outer
    ax.text(6.95, 4.85, "$x$\n(warm start)", ha="left", va="center",
            fontsize=8.2, color="#333")
    ax.text(5.0, 4.85, "one-way\ndependency", ha="center", va="center",
            fontsize=7.6, style="italic", color="#8a8f96")

    _arrow(ax, (5.0, 1.7), (5.0, 0.7), color=C_ACC)
    ax.text(5.0, 0.35, r"KKT point  $x^\star$", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=C_ACC)


def geometry(ax):
    ax.set_xlim(-0.5, 5.4)
    ax.set_ylim(-0.7, 4.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("B   Geometry: projection + multiplier $\\to$ KKT", fontsize=12,
                 loc="left", fontweight="bold")

    opt = np.array([4.3, 3.3])          # unconstrained objective optimum (outside)
    disk_c = np.array([1.6, 1.3])
    disk_r = 1.75

    # objective contours (dashed ellipses centred at the unconstrained optimum)
    for k in range(1, 6):
        ax.add_patch(Ellipse(opt, 1.5 * k, 1.15 * k, angle=25, fill=False,
                             ec=C_OBJ, ls=(0, (4, 3)), lw=1.0, zorder=1))
    ax.text(*(opt + [0.05, 0.0]), "$f$", color="#5b6470", fontsize=11,
            ha="center", va="center", fontweight="bold")
    _arrow(ax, (2.55, 1.55), (3.25, 2.05), color="#5b6470", lw=1.6, ms=11)

    # hard constraint  g(x) = x + y - 3.4 <= 0 ; infeasible half shaded
    infeas = Polygon([(0.6, 2.8), (5.4, -1.6), (5.4, 4.3), (1.0, 4.3)],
                     closed=True, fc=C_INFEAS, ec="none", alpha=0.6, zorder=0)
    ax.add_patch(infeas)
    xs = np.array([-0.5, 4.6])
    ax.plot(xs, 3.4 - xs, color=C_ACC, lw=2.0, zorder=3)
    ax.text(4.35, -0.35, r"$g(x)\leq 0$  (hard:", color=C_ACC, fontsize=9,
            ha="right", va="center")
    ax.text(4.35, -0.62, r"projection-free, multiplier $\lambda$)", color=C_ACC,
            fontsize=9, ha="right", va="center")
    ax.text(3.95, 3.75, "infeasible", color="#a03b3b", fontsize=8.5, style="italic",
            ha="center")

    # easy set S_easy (projectable disk)
    ax.add_patch(Circle(disk_c, disk_r, fc=C_EASY, ec="#5b8fc0", lw=1.6,
                        alpha=0.7, zorder=1))
    ax.text(0.55, 0.35, r"$S_{\mathrm{easy}}$", fontsize=12, color="#2c5f8f",
            ha="center", fontweight="bold")
    ax.text(0.95, -0.05, "(easy: projection)", fontsize=8, color="#2c5f8f", ha="center")

    # iterate trajectory: start -> projected steps -> onto active constraint
    traj = np.array([[0.35, 0.55], [1.05, 0.78], [1.7, 1.0], [2.05, 1.12], [2.2, 1.2]])
    ax.plot(traj[:, 0], traj[:, 1], ":", color="#444", lw=1.6, zorder=4)
    ax.scatter(traj[:-1, 0], traj[:-1, 1], s=22, color="#444", zorder=5)
    ax.scatter([traj[0, 0]], [traj[0, 1]], s=40, color="#444", zorder=6)
    ax.text(traj[0, 0] - 0.05, traj[0, 1] + 0.18, "start", fontsize=8, ha="left")

    # KKT point and the stationarity condition grad f = lambda grad g
    kkt = np.array([2.2, 1.2])
    ax.scatter(*kkt, s=140, marker="*", color=C_ACC, edgecolor="k", lw=0.5, zorder=7)
    n = np.array([1.0, 1.0]) / np.sqrt(2)
    _arrow(ax, tuple(kkt), tuple(kkt + 0.85 * n), color="#5b6470", lw=2.0, ms=13)
    _arrow(ax, tuple(kkt), tuple(kkt + 0.85 * n), color=C_ACC, lw=1.2, ms=10)
    ax.text(kkt[0] + 0.15, kkt[1] - 0.32,
            r"KKT point" "\n" r"$\nabla f = \lambda\,\nabla g$",
            fontsize=8.8, color=C_ACC, ha="left", va="top", fontweight="bold")


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   gridspec_kw={"width_ratios": [1.05, 1.0]})
    architecture(axA)
    geometry(axB)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "visual_abstract.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
