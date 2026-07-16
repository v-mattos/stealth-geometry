"""Budget-cap sweep (slide-10 evidence): how large can a STEALTHY attack be, and what
position impact does it cause, as a function of the disturbance bound Dbar?

For each Dbar we (i) solve the nominal robust tracking (5) -> g*, (ii) solve the attacker
best response (8) -> the worst stealthy constant actuation attack A0, and (iii) push that A0
through the true double integrator from rest (superposition) to get the realized position
deviation it can cause. The point: A0 (and hence the impact) is hard-capped by Dbar, so within
the paper's stated Dbar=0.01 a stealthy attack provably cannot reach the figure's ~0.5 excursion.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from deepc import FigConfig, build, simulate, m
from game import attacker_response
from tracking import nominal_tracking

STATED_DBAR = 0.01          # the paper's bound
PAPER_EXCURSION = 0.5       # apparent red excursion in Fig.1 (position units)


def impact_for_dbar(dbar: float):
    cfg = FigConfig(name=f"sweep", Tini=5, Tf=15, T=43, lambda_g=1.0,
                    delta_g=1e-4, Dbar=dbar, Dnorm=0.0, seed=1)
    d = build(cfg)
    g_star = nominal_tracking(d)["g_star"]
    atk = attacker_response(d, g_star)
    A0 = atk["A0"]
    # realized position deviation = double-integrator response to the constant attack A0
    Delta = simulate(np.full((m, cfg.Tf), A0), np.zeros(2))   # (p, Tf)
    impact = float(np.max(np.abs(Delta[0])))                  # max |Δ position|
    return abs(A0), impact


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    dbars = np.linspace(0.0, 0.05, 26)
    A0s, impacts = [], []
    for db in dbars:
        a0, imp = impact_for_dbar(db) if db > 0 else (0.0, 0.0)
        A0s.append(a0); impacts.append(imp)
    A0s, impacts = np.array(A0s), np.array(impacts)

    # value at the stated bound (for annotation)
    _, imp_stated = impact_for_dbar(STATED_DBAR)
    # bound needed to reach the paper's excursion (linear extrapolation)
    slope = impacts[-1] / dbars[-1]
    dbar_for_paper = PAPER_EXCURSION / slope

    red, gray, teal = "#be3e26", "0.45", "#1b96a6"
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(dbars, impacts, "-", color=red, lw=2.2, label="max stealthy attack impact")
    ax.axvline(STATED_DBAR, ls=":", color=teal, lw=1.5)
    ax.plot([STATED_DBAR], [imp_stated], "o", color=teal, ms=7, zorder=5)
    ax.annotate(f"stated $\\bar D$=0.01\n impact $\\approx${imp_stated:.2f}",
                xy=(STATED_DBAR, imp_stated), xytext=(0.014, imp_stated + 0.12),
                color=teal, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=teal, lw=1.2))
    ax.axhline(PAPER_EXCURSION, ls="--", color=gray, lw=1.3)
    ax.annotate(f"paper's visible red excursion ($\\approx$0.5)\n"
                f"would need $\\bar D\\approx${dbar_for_paper:.02f} "
                f"({dbar_for_paper/STATED_DBAR:.0f}$\\times$ the bound)",
                xy=(dbar_for_paper, PAPER_EXCURSION),
                xytext=(0.006, 0.60), color=gray, fontsize=8.5)
    ax.set_xlabel(r"disturbance bound $\bar D$")
    ax.set_ylabel(r"max position deviation from a stealthy attack")
    ax.set_title("A stealthy attack's impact is capped by the bound", fontsize=11)
    ax.set_xlim(0, 0.05); ax.set_ylim(0, 0.62); ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = os.path.join(here, "fig_budget.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"stated Dbar={STATED_DBAR}: A0={impact_for_dbar(STATED_DBAR)[0]:.2e}, "
          f"impact={imp_stated:.3f}; slope={slope:.2f}/unit; "
          f"Dbar for 0.5 excursion ~ {dbar_for_paper:.3f} -> {out}")


if __name__ == "__main__":
    main()
