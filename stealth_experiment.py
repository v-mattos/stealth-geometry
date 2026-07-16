from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from deepc import FIG1, build, simulate, m, n
from game import attacker_response
from stealth_geometry import StealthGeometry
from tracking import nominal_tracking

HERE = os.path.dirname(os.path.abspath(__file__))
RED, TEAL, GRAY, GOLD = "#be3e26", "#1b96a6", "0.45", "#dc9c2a"


def physical_impact(A0: float, Tf: int) -> float:
    """Push a constant actuation attack A0 through the true double integrator (rest ICs,
    superposition, matching sweep_budget.py); return max |position| deviation."""
    Delta = simulate(np.full((m, Tf), A0), np.zeros(2))
    return float(np.max(np.abs(Delta[0])))


def _direction_A1(sg: StealthGeometry) -> np.ndarray:
    v = sg.FAbar[:, 0].copy()                      # = [1_{n1};0_{n2};1_{n3}], exactly in R
    return v / np.linalg.norm(v)


def _direction_A2(sg: StealthGeometry, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(sg.n2)
    D = np.concatenate([np.zeros(sg.n1), w, np.zeros(sg.n3)])
    return D / np.linalg.norm(D)


def _direction_B(sg: StealthGeometry, seed: int) -> np.ndarray:
    """Unit direction with D2=0 and (D1,D3) orthogonal to FAbar's column-0 (zero-mean)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(sg.n1 + sg.n3)
    ones = np.ones(sg.n1 + sg.n3)
    z = z - (z @ ones) / (ones @ ones) * ones       # project out the (1,...,1) direction
    D1, D3 = z[:sg.n1], z[sg.n1:]
    D = np.concatenate([D1, np.zeros(sg.n2), D3])
    return D / np.linalg.norm(D)


def task2(cfg=FIG1, n_c: int = 21, seed: int = 0):
    d = build(cfg)
    sg = StealthGeometry.from_data(d)
    Dbar = cfg.Dbar
    cs = np.linspace(0.0, Dbar, n_c)

    dirs = {
        "A1: actuation bias ($\\in R$)": _direction_A1(sg),
        "A2: sensor bias ($\\in R$)": _direction_A2(sg, seed),
        "B: zero-mean actuation ($\\perp R$)": _direction_B(sg, seed + 1),
    }
    rows = []
    results = {}
    for label, dhat in dirs.items():
        impacts, rhos, dists = [], [], []
        for c in cs:
            D = c * dhat
            rho = sg.rho(D)
            dA0 = sg.max_marginal_actuation(D)      # = rho(D)/sqrt(n1+n3) (Prop. 1')
            imp = physical_impact(dA0, cfg.Tf)
            impacts.append(imp); rhos.append(rho); dists.append(sg.dist_to_R(D))
            rows.append((label, c, dists[-1], rho, imp))
        results[label] = dict(cs=cs, impact=np.array(impacts), rho=np.array(rhos),
                              dist=np.array(dists))

    # verify the closed-form predictions
    A1r = results["A1: actuation bias ($\\in R$)"]
    Br = results["B: zero-mean actuation ($\\perp R$)"]
    flat_err = float(np.max(np.abs(A1r["rho"] - Dbar)))
    decay_pred = np.sqrt(np.maximum(Dbar ** 2 - cs ** 2, 0.0))
    decay_err = float(np.max(np.abs(Br["rho"] - decay_pred)))

    colors = {"A1: actuation bias ($\\in R$)": TEAL,
             "A2: sensor bias ($\\in R$)": GOLD,
             "B: zero-mean actuation ($\\perp R$)": RED}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    # (left) physical impact A-vs-B
    for label, r in results.items():
        axL.plot(r["cs"], r["impact"], "o-", color=colors[label], ms=4, lw=1.8, label=label)
    axL.set_xlabel(r"disturbance magnitude $c=\|D\|$")
    axL.set_ylabel("physical impact (max |position| deviation)")
    axL.set_title("(a) marginal stealthy attack impact", fontsize=10)
    axL.grid(alpha=0.3); axL.legend(fontsize=8, loc="center left")
    # (right) the geometry that drives it: rho(D) (solid) and dist(D,R) (dashed)
    for label, r in results.items():
        axR.plot(r["cs"], r["rho"], "-", color=colors[label], lw=1.9)
        axR.plot(r["cs"], r["dist"], "--", color=colors[label], lw=1.2, alpha=0.7)
    axR.plot([], [], "k-", lw=1.9, label=r"$\rho(D)$ (stealth margin)")
    axR.plot([], [], "k--", lw=1.2, label=r"$\mathrm{dist}(D,R)$")
    axR.set_xlabel(r"disturbance magnitude $c=\|D\|$")
    axR.set_ylabel(r"$\rho(D)$, $\mathrm{dist}(D,R)$")
    axR.set_title(r"(b) stealth geometry: $\rho^2=\bar D^2-\mathrm{dist}(D,R)^2$", fontsize=10)
    axR.grid(alpha=0.3); axR.legend(fontsize=8, loc="center left")
    fig.suptitle(rf"Task 2 ({cfg.name}): attack impact is governed by $\mathrm{{dist}}(D,R)$, "
                 rf"$R=\mathrm{{range}}(\bar F_A)$, not by $\|D\|$", fontsize=10)
    tag = cfg.name.lower().replace(".", "")        # "Fig.1" -> "fig1"
    fig.tight_layout(); fig.savefig(os.path.join(HERE, f"fig_stealth_task2_{tag}.png"), dpi=150)
    plt.close(fig)

    print(f"Task 2 ({cfg.name}): family-A rho(D) flat, max|rho-Dbar|={flat_err:.2e} "
          f"(expect ~0); family-B rho(D) vs sqrt(Dbar^2-c^2) max err={decay_err:.2e}")
    print(f"  impact @ c=0: A1={A1r['impact'][0]:.4f} B={Br['impact'][0]:.4f} "
          f"(equal, sanity) | impact @ c=Dbar: A1={A1r['impact'][-1]:.4f} B={Br['impact'][-1]:.4f}")
    return dict(results=results, flat_err=flat_err, decay_err=decay_err, rows=rows, d=d, sg=sg)


# ---------------------------------------------------------------------------
# Task 3 -- Prop.-1 slack for the EXISTING (8) equilibrium attack, against several true D's
# ---------------------------------------------------------------------------
def task3(cfg=FIG1, n_random: int = 5, seed: int = 1):
    d = build(cfg)
    sg = StealthGeometry.from_data(d)
    g_star = nominal_tracking(d)["g_star"]
    atk = attacker_response(d, g_star)              # D-independent existing (8) attack
    n_a = sg.FAbar.shape[1]
    A_eq = np.concatenate([[atk["A0"]], atk["A_r"]])
    assert A_eq.shape == (n_a,), f"A_eq shape {A_eq.shape} != (n_a,)={n_a}"

    Dbar = cfg.Dbar
    candidates = {
        "A1 (c=Dbar)": Dbar * _direction_A1(sg),
        "A2 (c=Dbar)": Dbar * _direction_A2(sg, seed),
        "B (c=Dbar)": Dbar * _direction_B(sg, seed + 1),
        "D=0 (no disturb.)": np.zeros(sg.FAbar.shape[0]),
    }
    rng = np.random.default_rng(seed + 2)
    for k in range(n_random):
        raw = rng.standard_normal(sg.FAbar.shape[0])
        raw *= Dbar / np.linalg.norm(raw)           # random D with ||D||=Dbar (worst budget)
        candidates[f"random {k}"] = raw

    A0_eq = atk["A0"]
    # For a FIXED concrete attack the correct test is ellipsoid MEMBERSHIP using the real A_r
    # (Prop. 1). The A0-shadow ignores A_r and is only necessary. Slacks are in squared units
    # (rho^2 - LHS) so shadow and ellipsoid are directly comparable; LHS_ell = LHS_shadow +
    # ||A_r - A_r_c||^2, so the ellipsoid can only be stricter.
    print(f"\nTask 3 ({cfg.name}): existing (8) attack (A0={A0_eq:.3e}, ||A_r||="
          f"{np.linalg.norm(atk['A_r']):.3e}) tested against several true D at ||D||={Dbar}."
          f" Slacks in squared units; <0 = not stealthy vs that D.")
    print(f"  {'D':<18}{'dist(D,R)':>11}{'rho^2':>11}{'slack_shadow':>14}{'slack_ellip':>13}"
          f"{'LHS_ell/rho^2':>14}")
    rows = []
    for label, D in candidates.items():
        dist = sg.dist_to_R(D)
        rho2 = sg.rho(D) ** 2
        _, slack_sh_lin = sg.shadow_check(A0_eq, D)          # linear-unit slack
        lhs_sh = sg.shadow_check(A0_eq, D)[0]
        slack_sh = rho2 - lhs_sh ** 2                        # convert shadow to squared units
        lhs_ell, slack_ell = sg.membership_ellipsoid(A_eq, D)
        ratio = lhs_ell / rho2 if rho2 > 0 else float("inf")
        rows.append((label, dist, rho2, slack_sh, slack_ell, ratio))
        print(f"  {label:<18}{dist:>11.3e}{rho2:>11.3e}{slack_sh:>14.3e}{slack_ell:>13.3e}"
              f"{ratio:>14.2f}")
    return dict(A_eq=A_eq, rows=rows, d=d, sg=sg)


# ---------------------------------------------------------------------------
# Summary table (deliverable): c, dist(D,R), rho(D), marginal impact, Prop.1' slack
# ---------------------------------------------------------------------------
def summary_table(cfg=FIG1, n: int = 6, seed: int = 0):
    d = build(cfg)
    sg = StealthGeometry.from_data(d)
    g_star = nominal_tracking(d)["g_star"]
    atk = attacker_response(d, g_star)
    A_eq = np.concatenate([[atk["A0"]], atk["A_r"]])
    fams = {"A1 (in R)": _direction_A1(sg), "B (perp R)": _direction_B(sg, seed + 1)}
    cs = np.linspace(0.0, cfg.Dbar, n)
    print(f"\nSummary table ({cfg.name}); ellipsoid slack = rho^2 - ||FAbar(A_eq - A_c)||^2:")
    print(f"  {'family':<11}{'c':>10}{'dist(D,R)':>12}{'rho(D)':>10}{'impact':>10}{'slack_ell':>12}")
    for label, dhat in fams.items():
        for c in cs:
            D = c * dhat
            imp = physical_impact(sg.max_marginal_actuation(D), cfg.Tf)
            _, slack = sg.membership_ellipsoid(A_eq, D)
            print(f"  {label:<11}{c:>10.4f}{sg.dist_to_R(D):>12.2e}{sg.rho(D):>10.2e}"
                  f"{imp:>10.4f}{slack:>12.2e}")

def task3_validation(cfg=FIG1, n_c: int = 21, seed: int = 0):
    d = build(cfg)
    sg = StealthGeometry.from_data(d)
    g_star = nominal_tracking(d)["g_star"]
    yref = d.y_ref
    y_nom = d.Yf @ g_star                       # defender's future output (the paper's curve)
    # Attack's physical effect is, by superposition, A0 * (plant response to a unit constant
    # actuation), MEASURED here by an actual double-integrator rollout -- independent of the
    # baseline state, so it is a genuine plant simulation, not the analytic formula.
    unit = simulate(np.ones((m, cfg.Tf)), np.zeros(n)).T.ravel()
    gamma = float(np.max(np.abs(unit[0::2])))   # peak |position| response (T4)

    dirs = {"A1 ($\\in R$)": _direction_A1(sg),
            "A2 ($\\in R$)": _direction_A2(sg, seed),
            "B ($\\perp R$)": _direction_B(sg, seed + 1)}
    cs = np.linspace(0.0, cfg.Dbar, n_c)
    results = {}
    z_asym = None
    for label, dhat in dirs.items():
        pred, meas = [], []
        for c in cs:
            D = c * dhat
            h = sg.max_marginal_actuation(D)            # rho/sqrt(n1+n3): A0 semi-axis
            A0c = sg.A0_center(D)
            # best-response over the ellipsoid: A_r = A_r_c, A0 at the endpoint maximizing the
            # realized tracking error of the rolled-out output y_nom + A0*unit.
            y_end = {A0: y_nom + A0 * unit for A0 in (A0c - h, A0c + h)}
            errs = {A0: float(np.linalg.norm(y - yref)) for A0, y in y_end.items()}
            A0star = max(errs, key=errs.get)
            # measured impact: peak |position| deviation vs the A0_c baseline (attack beyond D)
            dev = (y_end[A0star] - (y_nom + A0c * unit))[0::2]
            meas.append(float(np.max(np.abs(dev))))
            pred.append(gamma * h)                       # eq. (11): I(D) = gamma * rho/sqrt(n1+n3)
            if c == cs[-1] and label.startswith("A1"):
                z_asym = (errs[A0c - h], errs[A0c + h])  # asymmetry from yref=0.5
        results[label] = (cs, np.array(pred), np.array(meas))

    rel_err = max(float(np.max(np.abs(p - mmeas))) for _, p, mmeas in results.values())
    print(f"\nT3 validation ({cfg.name}): gamma(measured)={gamma:.4f}, "
          f"max |I_pred - dev_measured| = {rel_err:.2e} over all families/c.")
    if z_asym:
        print(f"  realized tracking error at the two A0 endpoints (A1, c=Dbar): "
              f"{z_asym[0]:.4f} vs {z_asym[1]:.4f} (asymmetric: yref=0.5 is not centered).")

    colors = {"A1 ($\\in R$)": TEAL, "A2 ($\\in R$)": GOLD, "B ($\\perp R$)": RED}
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for label, (csx, pred, mvals) in results.items():
        ax.plot(csx, pred, "-", color=colors[label], lw=1.8,
                label=f"{label}: $I(D)$ predicted")
        ax.plot(csx, mvals, "o", color=colors[label], ms=4, mfc="none",
                label=f"{label}: measured")
    ax.set_xlabel(r"disturbance magnitude $c=\|D\|$")
    ax.set_ylabel("peak position deviation from attack")
    ax.set_title(rf"T3 ({cfg.name}): best-response $\to$ plant vs. prediction (11)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7.5, ncol=1, loc="center left")
    tag = cfg.name.lower().replace(".", "")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, f"fig_stealth_val_{tag}.png"), dpi=150)
    plt.close(fig)
    return dict(gamma=gamma, rel_err=rel_err, results=results)


def widened_halfwidth(d, sg, D):
    """Return (h0, h1): the base actuation half-width h0 = rho(D)/sqrt(n1+n3) and the widened
    one h1 obtained by admitting the coefficient freedom ||g~-g|| <= delta_g. h1 is the largest
    |A0 - A0_c| with min_{||z||<=delta_g} || 1_13 A0 + D13 - [Up;Uf] z || <= Dbar."""
    from scipy.optimize import minimize, NonlinearConstraint

    n1, n3, n_g = sg.n1, sg.n3, d.cfg.n_g
    dg, Dbar = d.cfg.delta_g, d.cfg.Dbar
    M = np.vstack([d.Up, d.Uf])
    ones13 = np.concatenate([np.ones(n1), np.ones(n3)])
    idx13 = np.r_[0:n1, n1 + sg.n2:n1 + sg.n2 + n3]
    D13 = D[idx13]
    A0c = sg.A0_center(D)
    h0 = sg.max_marginal_actuation(D)

    def r_of_A0(A0):
        v = ones13 * A0 + D13
        obj = lambda z: float((v - M @ z) @ (v - M @ z))
        con = NonlinearConstraint(lambda z: z @ z, 0.0, dg ** 2)
        res = minimize(obj, np.zeros(n_g), constraints=[con], method="SLSQP",
                       options={"maxiter": 500, "ftol": 1e-20})
        return float(np.sqrt(max(res.fun, 0.0)))

    lo, hi = h0, h0 + 5.0 * float(np.linalg.svd(M, compute_uv=False)[0]) * dg + 1e-9
    while r_of_A0(A0c + hi) < Dbar:
        hi *= 1.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if r_of_A0(A0c + mid) <= Dbar:
            lo = mid
        else:
            hi = mid
    return h0, 0.5 * (lo + hi)


def delta_g_widening(cfg=FIG1, seed: int = 0):
    d = build(cfg)
    sg = StealthGeometry.from_data(d)
    n1, n3 = sg.n1, sg.n3
    dg, Dbar = d.cfg.delta_g, d.cfg.Dbar
    smax_M = float(np.linalg.svd(np.vstack([d.Up, d.Uf]), compute_uv=False)[0])
    predicted = smax_M * dg                           # Remark-2 rho-level widening estimate

    cases = {
        "D=0": np.zeros(sg.FAbar.shape[0]),
        "D in R (||D||=Dbar)": Dbar * _direction_A1(sg),
        "D perp R (||D||=Dbar)": Dbar * _direction_B(sg, seed + 1),
    }
    print(f"\nT4 delta_g widening ({cfg.name}): sigma_max([Up;Uf])={smax_M:.3f}, "
          f"delta_g={dg}, predicted rho-widening ~ sigma_max*delta_g = {predicted:.3e}")
    print(f"  {'D':<24}{'h0=rho/sqrt(n)':>16}{'h1 (with g~)':>15}{'rho-widen (meas)':>18}"
          f"{'meas/pred':>11}")
    out = {}
    for label, D in cases.items():
        h0, h1 = widened_halfwidth(d, sg, D)
        widen_rho = (h1 - h0) * np.sqrt(n1 + n3)      # convert half-width widening to rho units
        out[label] = widen_rho
        print(f"  {label:<24}{h0:>16.4e}{h1:>15.4e}{widen_rho:>18.4e}"
              f"{widen_rho / predicted:>11.2f}")
    return dict(predicted=predicted, smax_M=smax_M, widen=out)


if __name__ == "__main__":
    from deepc import FIG2
    for cfg in (FIG1, FIG2):
        task2(cfg)
        task3(cfg)
        summary_table(cfg)
        task3_validation(cfg)
        delta_g_widening(cfg)
