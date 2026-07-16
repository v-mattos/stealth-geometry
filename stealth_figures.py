"""Figures for the stealth-geometry note (T1-T3 of the review response). Consumes the
geometry of stealth_geometry.py and the reproduction's g_star/g_hat; does NOT modify the
equilibrium reproduction.

  T1  fig_stealth_time.png   -- opening, time-domain: two panels D in R vs D perp R at the
                                SAME ||D||=Dbar, attack-free vs worst-stealthy in position and
                                velocity, sharing the measured past (t<=Tini). (Robust g_hat is
                                omitted: not this note's contribution; see the security-vs-Nash
                                remark in the text.)
  T1b fig_stealth_deviation.png -- the attack-INDUCED deviation Delta y = y_atk - y_af, which
                                isolates the ~0.10 effect the absolute time view buries; panel
                                (b) also shows the delta_g-freedom residual (~0.03).
  T2  fig_stealth_angle.png  -- angle sweep at fixed ||D||=Dbar: impact vs theta=angle(D,R)
                                for Cfg-1/2, and the two curves normalized collapse onto
                                cos(theta) (geometric, not dynamic, difference).
  T3  fig_stealth_ellipsoid.png -- the stealthy ellipsoid in the (A0,||A_r||) plane for three
                                disturbances, with [1]'s fixed equilibrium attack (outside all)
                                and the necessary-only A0-shadow band.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from deepc import FIG1, FIG2, build, simulate, m, n
from game import attacker_response
from stealth_geometry import StealthGeometry
from stealth_experiment import _direction_A1, _direction_B, _direction_A2, widened_halfwidth
from tracking import nominal_tracking

HERE = os.path.dirname(os.path.abspath(__file__))
GRAY, RED, TEAL, GOLD = "0.45", "#be3e26", "#1b96a6", "#dc9c2a"


def _unit_response(Tf):
    """Position/velocity response to a unit constant actuation, as (Tf, 2)."""
    return simulate(np.ones((m, Tf)), np.zeros(n)).T


# ---------------------------------------------------------------------------
# T1 -- time-domain opening figure
# ---------------------------------------------------------------------------
def t1_timedomain(cfg=FIG1, seed: int = 0):
    d = build(cfg)
    sg = StealthGeometry.from_data(d)
    Tf, Tini, Dbar = cfg.Tf, cfg.Tini, cfg.Dbar
    g_star = nominal_tracking(d)["g_star"]
    # Follow the reproduction (plots.py): every curve shows the SHARED measured initial
    # trajectory y_ini in the past (t<=Tini) then its own future prediction Yf g. (The robust
    # policy g_hat is deliberately omitted -- it is not this note's contribution and its
    # isolated-(9) output initial condition is vacuous; see the security-vs-Nash remark in the
    # text. This figure shows only the attack-vs-no-attack contrast.)
    past = d.y_ini.reshape(Tini, 2)

    def full(g_future, extra=None):
        fut = (d.Yf @ g_future).reshape(Tf, 2)
        if extra is not None:
            fut = fut + extra
        return np.vstack([past, fut])                 # (Tini+Tf, 2)

    yref = d.y_ref.reshape(Tf, 2)
    unit = _unit_response(Tf)                          # (Tf, 2), attack effect (future only)
    y_af = full(g_star)
    t = np.arange(1, Tini + Tf + 1)

    panels = {r"(a) $D\in R$": Dbar * _direction_A1(sg),
              r"(b) $D\perp R$": Dbar * _direction_B(sg, seed + 1)}
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.4), sharex=True)
    for i, (label, D) in enumerate(panels.items()):
        h = sg.max_marginal_actuation(D)              # rho/sqrt(n1+n3): stealthy authority
        rho, dist = sg.rho(D), sg.dist_to_R(D)
        yf_fut = (d.Yf @ g_star).reshape(Tf, 2)
        s = max((-1.0, 1.0), key=lambda s: np.linalg.norm((yf_fut + s * h * unit) - yref))
        y_atk = full(g_star, extra=s * h * unit)       # attack acts only in the future
        for j, (st, name) in enumerate([(0, "position"), (1, "velocity")]):
            ax = axes[i, j]
            ax.axvline(Tini, ls="--", color="0.6", lw=0.9)
            ax.plot(t, y_af[:, st], color=GRAY, lw=1.8, label="attack-free $g^\\star$")
            ax.plot(t, y_atk[:, st], color=RED, lw=1.8, label="worst stealthy")
            ax.axhline(yref[0, st], color="k", lw=0.6, ls=":")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(name, fontsize=10)
            if i == 1:
                ax.set_xlabel(r"time step $t$ (shared past $\to$ future at $T_{\mathrm{ini}}$)")
        axes[i, 0].text(0.03, 0.03,
                        rf"$\|D\|={Dbar}$, $\mathrm{{dist}}(D,R)={dist:.3f}$, $\rho={rho:.3f}$",
                        transform=axes[i, 0].transAxes, fontsize=8, va="bottom",
                        bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        axes[i, 0].annotate(label, xy=(-0.28, 0.5), xycoords="axes fraction",
                            rotation=90, va="center", fontsize=10, fontweight="bold")
        if i == 1:  # D perp R: the rho=0 collapse assumes g~=g (see the delta_g section)
            axes[1, 1].text(0.5, 0.06,
                            r"$\tilde g{=}g$ idealization; with $\delta_g{=}10^{-4}$"
                            "\n"
                            r"residual ${\approx}0.03$ (Sec.~V-D)",
                            transform=axes[1, 1].transAxes, fontsize=6.5, ha="center",
                            va="bottom", color=RED,
                            bbox=dict(boxstyle="round", fc="white", ec=RED, alpha=0.8))
    axes[0, 1].legend(fontsize=7.5, loc="best")
    fig.suptitle(r"Same $\|D\|=\bar D$, opposite orientation (shared past, then future)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_stealth_time.png"), dpi=150)
    plt.close(fig)
    print(f"T1 ({cfg.name}): panel (a) peak |dev|={sg.max_marginal_actuation(panels[r'(a) $D\in R$'])*np.max(np.abs(unit[:,0])):.4f}, "
          f"panel (b) peak |dev|={sg.max_marginal_actuation(panels[r'(b) $D\perp R$'])*np.max(np.abs(unit[:,0])):.4f}")


# ---------------------------------------------------------------------------
# T1b -- attack-induced DEVIATION (isolates the thesis the time figure buries)
# ---------------------------------------------------------------------------
def t1_deviation(cfg=FIG1, seed: int = 0):
    d = build(cfg)
    sg = StealthGeometry.from_data(d)
    Tf, Dbar = cfg.Tf, cfg.Dbar
    unit = _unit_response(Tf)                          # (Tf, 2): the attack's per-A0 effect
    gamma = float(np.max(np.abs(unit[:, 0])))
    t = np.arange(Tf)
    panels = {r"(a) $D\in R$": Dbar * _direction_A1(sg),
              r"(b) $D\perp R$": Dbar * _direction_B(sg, seed + 1)}
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.0), sharex=True)
    for i, (label, D) in enumerate(panels.items()):
        h = sg.max_marginal_actuation(D)              # base (g~=g) actuation authority
        rho, dist = sg.rho(D), sg.dist_to_R(D)
        dev = h * unit                                # Delta y(t) = y_atk - y_af = h * unit
        peak = float(np.max(np.abs(dev[:, 0])))
        for j, name in enumerate(["position", "velocity"]):
            ax = axes[i, j]
            ax.axhline(0, color="0.7", lw=0.6)
            ax.plot(t, dev[:, j], color=RED, lw=1.9, label=r"$\Delta y$ (attack $-$ attack-free)")
            if i == 1:  # D perp R: add the g~-freedom residual (Sec. delta_g)
                _, h1 = widened_halfwidth(d, sg, D)
                ax.plot(t, h1 * unit[:, j], color=GOLD, lw=1.6, ls="--",
                        label=r"with $\delta_g$ freedom ($h_1$)")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(name, fontsize=10)
            if i == 1:
                ax.set_xlabel("time step $t$")
        axes[i, 0].set_ylabel("position deviation" if i == 0 else "position deviation")
        axes[i, 1].set_ylabel("velocity deviation")
        axes[i, 0].annotate(label, xy=(-0.32, 0.5), xycoords="axes fraction",
                            rotation=90, va="center", fontsize=10, fontweight="bold")
        # place the info box in the free corner of each panel (avoids the legend)
        tx, ty, va, ha = (0.03, 0.95, "top", "left") if i == 0 else (0.97, 0.05, "bottom", "right")
        axes[i, 0].text(tx, ty,
                        rf"$\|D\|={Dbar}$, $\mathrm{{dist}}={dist:.3f}$, $\rho={rho:.3f}$"
                        "\n" rf"peak $|\Delta y|={peak:.4f}$",
                        transform=axes[i, 0].transAxes, fontsize=7.5, va=va, ha=ha,
                        bbox=dict(boxstyle="round", fc="white", alpha=0.75))
    axes[0, 0].legend(fontsize=7, loc="lower right")
    axes[1, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle(r"Attack-induced deviation $\Delta y(t)$: the thesis the absolute view buries",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_stealth_deviation.png"), dpi=150)
    plt.close(fig)
    predicted_peak = gamma * sg.max_marginal_actuation(panels[r"(a) $D\in R$"])
    _, h1b = widened_halfwidth(d, sg, panels[r"(b) $D\perp R$"])
    print(f"T1b deviation ({cfg.name}): (a) peak={gamma*sg.max_marginal_actuation(panels[r'(a) $D\in R$']):.4f} "
          f"(= I(D)={predicted_peak:.4f}); (b) base peak=0, delta_g-residual peak={gamma*h1b:.4f}")


# ---------------------------------------------------------------------------
# T2 -- angle sweep + normalized collapse
# ---------------------------------------------------------------------------
def t2_anglesweep(n_theta: int = 46, seed: int = 0):
    thetas = np.linspace(0.0, np.pi / 2, n_theta)
    curves = {}
    dist_err = 0.0
    for cfg in (FIG1, FIG2):
        d = build(cfg)
        sg = StealthGeometry.from_data(d)
        Dbar, n1n3 = cfg.Dbar, sg.n1 + sg.n3
        gamma = float(np.max(np.abs(_unit_response(cfg.Tf)[:, 0])))
        uR = sg.FAbar[:, 0] / np.linalg.norm(sg.FAbar[:, 0])   # unit in R (actuation column)
        uP = _direction_B(sg, seed + 1)                        # unit in R^perp (zero-mean)
        imp = []
        for th in thetas:
            D = Dbar * (np.cos(th) * uR + np.sin(th) * uP)
            dist_err = max(dist_err, abs(sg.dist_to_R(D) - Dbar * np.sin(th)))
            imp.append(gamma * sg.max_marginal_actuation(D))   # = gamma*rho/sqrt(n1+n3)
        norm_const = gamma * Dbar / np.sqrt(n1n3)
        curves[cfg.name] = (np.array(imp), norm_const, gamma, n1n3)

    collapse_err = 0.0
    for name, (imp, nc, _, _) in curves.items():
        collapse_err = max(collapse_err, float(np.max(np.abs(imp / nc - np.cos(thetas)))))

    deg = np.degrees(thetas)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.9))
    cols = {"Fig.1": TEAL, "Fig.2": RED}
    for name, (imp, nc, gamma, n1n3) in curves.items():
        axL.plot(deg, imp, "-", color=cols[name], lw=1.9,
                 label=f"{name} ($n_1{{+}}n_3={n1n3}$)")
        axR.plot(deg, imp / nc, "o", color=cols[name], ms=3, mfc="none",
                 label=f"{name}")
    axR.plot(deg, np.cos(thetas), "k-", lw=1.2, label=r"$\cos\theta$")
    axL.set_xlabel(r"orientation $\theta=\angle(D,R)$ (deg)")
    axL.set_ylabel("stealthy attack impact")
    axL.set_title(r"(a) impact vs. orientation ($\|D\|=\bar D$ fixed)", fontsize=10)
    axL.grid(alpha=0.3); axL.legend(fontsize=8)
    axR.set_xlabel(r"orientation $\theta$ (deg)")
    axR.set_ylabel(r"impact $/\,(\gamma\bar D/\sqrt{n_1+n_3})$")
    axR.set_title(r"(b) normalized curves collapse onto $\cos\theta$", fontsize=10)
    axR.grid(alpha=0.3); axR.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_stealth_angle.png"), dpi=150)
    plt.close(fig)
    print(f"T2: dist(D,R)=Dbar*sin(theta) to {dist_err:.2e}; normalized-collapse onto cos(theta) "
          f"error = {collapse_err:.2e}; ratio of flat impacts sqrt(24/20)={np.sqrt(24/20):.4f}")


# ---------------------------------------------------------------------------
# T3 -- the stealthy ellipsoid in the (A0, ||A_r||) plane
# ---------------------------------------------------------------------------
def t3_ellipsoid(cfg=FIG1, seed: int = 0):
    d = build(cfg)
    sg = StealthGeometry.from_data(d)
    Dbar, n1n3 = cfg.Dbar, sg.n1 + sg.n3
    sq = np.sqrt(n1n3)
    g_star = nominal_tracking(d)["g_star"]
    atk = attacker_response(d, g_star)
    A0_eq, Ar_eq = atk["A0"], float(np.linalg.norm(atk["A_r"]))

    cases = [("$D=0$", np.zeros(sg.FAbar.shape[0]), TEAL),
             (r"$D\in R,\ \|D\|=\bar D$", Dbar * _direction_A1(sg), GOLD),
             (r"$D\perp R,\ \|D\|=\bar D$", Dbar * _direction_B(sg, seed + 1), RED)]

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    phi = np.linspace(0.0, np.pi, 240)                # upper half (||A_r|| >= 0)
    for label, D, col in cases:
        A0c, rho = sg.A0_center(D), sg.rho(D)
        if rho > 1e-12:
            ax.plot(A0c + (rho / sq) * np.cos(phi), rho * np.sin(phi), "-", color=col, lw=2.0,
                    label=label)
            ax.plot([A0c], [0], ".", color=col, ms=6)
        else:
            ax.plot([A0c], [0], "x", color=col, ms=11, mew=2.5,
                    label=label + r" ($\rho{=}0$, point)")
    # A0-shadow band for D=0 (necessary-only): |A0| <= rho/sqrt(n1+n3) = Dbar/sqrt(n1+n3)
    ax.axvspan(-Dbar / sq, Dbar / sq, color=TEAL, alpha=0.08)
    ax.text(0.0, Dbar * 0.40, r"$A_0$-shadow" "\n" r"($D{=}0$)", ha="center", va="center",
            fontsize=7.5, color="#127885")
    # [1]'s fixed equilibrium attack -- outside every ellipsoid
    ax.plot([A0_eq], [Ar_eq], "*", color="k", ms=15, zorder=5,
            label=r"[1] equilibrium attack")
    ax.annotate(rf"$(A_0,\|A_r\|)=({A0_eq:.1e},{Ar_eq:.1e})$", xy=(A0_eq, Ar_eq),
                xytext=(A0_eq + 0.0009, Ar_eq + 0.0012), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.axhline(0, color="0.7", lw=0.6)
    ax.set_xlabel(r"actuation attack $A_0$")
    ax.set_ylabel(r"deception magnitude $\|A_r\|$")
    # (no internal title; the LaTeX caption carries the description)
    ax.set_ylim(-0.0008, Dbar * 1.5); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_stealth_ellipsoid.png"), dpi=150)
    plt.close(fig)
    print(f"T3 ({cfg.name}): equilibrium attack (A0={A0_eq:.2e}, ||Ar||={Ar_eq:.2e}); "
          f"D=0 ellipsoid semi-axes: A0={Dbar/sq:.2e}, ||Ar||={Dbar:.2e} -> attack is outside.")


if __name__ == "__main__":
    t1_timedomain()
    t1_deviation()
    t2_anglesweep()
    t3_ellipsoid()
