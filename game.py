"""The two-player game: attacker's best response (8) and defender's security policy (9).

Both are min-max / max problems whose inner part maximizes a CONVEX quadratic over the
intersection of norm balls and an affine subspace (nonconvex), but in LOW dimension with
the example injection structure. We exploit that structure:

  Fa1 = 1_{n1} e1^T,  Fa2 = [0 I_{n2}],  Fa3 = 1_{n3} e1^T,  e1 in R^{n2+1}.
  So an attack A = [A0 ; A_r] (scalar A0 + vector A_r in R^{n2}) injects:
     - A0 (a CONSTANT) into every initial-input row (via Fa1) and every future-input row
       (via Fa3)  -> a constant actuation attack,
     - A_r into the initial-output rows (via Fa2)        -> a deception attack on y_ini.

We solve each best response with scipy (SLSQP) from informed warm starts. The defender
problem (9) is a nested min-max; the attacker problem (8) is a single max.

"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from deepc import DeePCData


# ---------------------------------------------------------------------------
# (9)  Defender's security policy  ->  g_hat  (BLUE)
#   min_{g_hat}  max_{A_hat, D_hat}  ||Yf g_hat - yref||^2
#                                  + ||Uf g_hat + Fd3 D_hat + Fa3 A_hat||^2
#                                  + lambda_g ||g_hat||^2
#   s.t.  [u_ini; y_ini] = [Up;Yp] g_hat + [Fd1;Fd2] D_hat + [Fa1;Fa2] A_hat,
#         ||D_hat|| <= Dbar.
#
# Inner reduction (example injection): write D=[Da;Db;Dc], A=[A0;Ar].
#   init-input  (n1):  Da + 1_{n1} A0 = u_ini - Up g_hat = rhs_u   -> Da = rhs_u - 1 A0
#   init-output (n2):  Db + Ar        = y_ini - Yp g_hat = rhs_y   -> Ar = rhs_y - Db (free)
#   cost term:         ||Uf g_hat + Dc + 1_{n3} A0||^2
#   budget:            ||Da||^2 + ||Db||^2 + ||Dc||^2 <= Dbar^2.
# Db only wastes budget -> Db*=0. Inner max over (A0 scalar, Dc in R^n3):
#   max ||Uf g_hat + Dc + 1_{n3} A0||^2
#     s.t. ||rhs_u - 1_{n1} A0||^2 + ||Dc||^2 <= Dbar^2.
# ---------------------------------------------------------------------------
def _defender_inner_max(g_hat, d: DeePCData):
    """Worst-case (A0, Dc) and the achieved ||Uf g + Fd3 D + Fa3 A||^2.

    Closed-form reduction: for fixed A0 the max over Dc on a ball is
        inner(A0) = ( ||w + 1_{n3} A0|| + rho(A0) )^2 ,
        rho(A0)^2 = Dbar^2 - ||rhs_u - 1_{n1} A0||^2   (>= 0 required),
    so the whole inner max is a 1-D maximization over the scalar actuation attack A0,
    on the interval where rho^2 >= 0. We grid + golden-refine. Returns (value,A0,Dc,Da)."""
    cfg = d.cfg
    w = d.Uf @ g_hat                                   # n3 vector
    rhs_u = d.u_ini - d.Up @ g_hat                     # n1 vector
    n1, n3 = cfg.n1, cfg.n3
    Dbar2 = cfg.Dbar ** 2
    s_sum = float(np.sum(rhs_u)); s_sq = float(rhs_u @ rhs_u)

    def rho2(A0):
        return Dbar2 - (s_sq - 2.0 * A0 * s_sum + n1 * A0 * A0)

    # feasible A0 interval: rho2(A0) >= 0  <=> n1 A0^2 - 2 s_sum A0 + (s_sq - Dbar^2) <= 0
    a, b, c = n1, -2.0 * s_sum, (s_sq - Dbar2)
    disc = b * b - 4 * a * c
    if disc < 0:                                       # infeasible (||rhs_u|| > Dbar): clamp
        A0_lo = A0_hi = s_sum / n1 if n1 else 0.0
    else:
        A0_lo = (-b - np.sqrt(disc)) / (2 * a)
        A0_hi = (-b + np.sqrt(disc)) / (2 * a)

    def inner(A0):
        r2 = max(rho2(A0), 0.0)
        return (np.linalg.norm(w + A0) + np.sqrt(r2)) ** 2

    grid = np.linspace(A0_lo, A0_hi, 41)
    vals = [inner(x) for x in grid]
    k = int(np.argmax(vals))
    lo = grid[max(k - 1, 0)]; hi = grid[min(k + 1, len(grid) - 1)]
    # golden-section refine on [lo,hi]
    gr = (np.sqrt(5) - 1) / 2
    x1 = hi - gr * (hi - lo); x2 = lo + gr * (hi - lo)
    f1, f2 = inner(x1), inner(x2)
    for _ in range(60):
        if f1 < f2:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + gr * (hi - lo); f2 = inner(x2)
        else:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - gr * (hi - lo); f1 = inner(x1)
    A0 = 0.5 * (lo + hi)
    r = np.sqrt(max(rho2(A0), 0.0))
    wv = w + A0
    nw = np.linalg.norm(wv)
    Dc = (r * wv / nw) if nw > 1e-12 else np.zeros(n3)
    Da = rhs_u - A0
    return inner(A0), A0, Dc, Da


def defender_security(d: DeePCData, warm: np.ndarray | None = None):
    """Solve (9). Returns g_hat (BLUE) and the worst-case attack it hedges against.

    Note on the initial condition: (9) lets the (unbounded) worst-case deception attack A_hat
    explain the manipulated initial measurements, so the security policy g_hat does NOT bind to
    the (potentially corrupted) initial trajectory -- it ignores the manipulable measurements
    and tracks y_ref robustly. This is faithful to (9) as written and gives blue as the clearly
    best tracker (the paper's core message). In the paper's figure blue still SHARES the past
    (the past is measured history, common to all players); we plot it as such (see plots.py).
    Reproducing the paper's exact blue *future* shape (a damped ramp out of the dip rather than
    an immediate jump to y_ref) would need the full equilibrium's bounded A_hat, which the paper
    does not specify -- documented in the README."""
    cfg = d.cfg

    def value(g_hat):
        track = d.Yf @ g_hat - d.y_ref
        inner, *_ = _defender_inner_max(g_hat, d)
        return track @ track + inner + cfg.lambda_g * (g_hat @ g_hat)

    if warm is None:
        Aq = d.Yf.T @ d.Yf + d.Uf.T @ d.Uf + cfg.lambda_g * np.eye(cfg.n_g)
        warm = np.linalg.solve(Aq, d.Yf.T @ d.y_ref)

    res = minimize(value, warm, method="L-BFGS-B",
                   options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9})
    g_hat = res.x
    inner, A0, Dc, Da = _defender_inner_max(g_hat, d)
    return {"g_hat": g_hat, "y_pred": d.Yf @ g_hat, "u_nom": d.Uf @ g_hat,
            "value": res.fun, "ok": res.success, "A0": A0,
            "y_pred_track_err": np.linalg.norm(d.Yf @ g_hat - d.y_ref)}


# ---------------------------------------------------------------------------
# (8)  Attacker's best response  ->  g_tilde (RED), attack A
#   max_{g_tilde, D_tilde, A}  ||Yf g_tilde - yref||^2 + ||Uf g_tilde + Fd3 D_tilde||^2
#   s.t.  ||g* - g_tilde|| <= delta_g,  ||D_tilde|| <= Dbar,
#         [u_ini;y_ini;u] + [Fa1;Fa2;Fa3] A = [Up;Yp;Uf] g_tilde + [Fd1;Fd2;Fd3] D_tilde.
#
# Given g* (nominal, eq.5) and the defender's applied input u (= Uf g*, defender plays
# nominal). The equality DEFINES the attack A and disturbance from g_tilde:
#   future (n3):  u + 1_{n3} A0 = Uf g_tilde + Dc   ->  Dc = u - Uf g_tilde + 1_{n3} A0
#   init-input(n1): u_ini + 1_{n1} A0 = Up g_tilde + Da -> Da = u_ini - Up g_tilde + 1 A0
#   init-output(n2): y_ini + Ar = Yp g_tilde + Db   ->  Ar = Yp g_tilde + Db - y_ini (free)
# Free decision vars: g_tilde (||g*-g_tilde||<=delta_g), A0 (scalar), Db (n2), Dc-as-Dtilde.
# We optimize over (g_tilde, A0, Db, Dc) with ||g*-g_tilde||<=delta_g, ||D_tilde||<=Dbar.
# ---------------------------------------------------------------------------
def attacker_response(d: DeePCData, g_star: np.ndarray, u_applied: np.ndarray | None = None):
    """Solve (8). Returns g_tilde (RED), the attack (A0, A_r), and the effective input.

    Clean reduction (see module docstring). Substituting the equality constraints:
      Da = A0 - Up g_t,   Dc = u + 1_{n3} A0 - Uf g_t,   so  Uf g_t + Dc = u + 1_{n3} A0,
    i.e. the future input under attack is just u + a CONSTANT actuation attack A0,
    independent of g_t. With g_t = g* + delta_g * v (||v||<=1) the problem is

      max_{||v||<=1, A0}  ||Yf g* + delta_g Yf v - yref||^2 + ||u + 1_{n3} A0||^2
        s.t. ||A0 - Up(g*+delta_g v)||^2 + ||(u+1 A0) - Uf(g*+delta_g v)||^2 <= Dbar^2,
    a well-scaled max of a quadratic over unit ball + one quadratic constraint (Db*=0)."""
    cfg = d.cfg
    n_g, n3 = cfg.n_g, cfg.n3
    u = d.Uf @ g_star if u_applied is None else u_applied
    Dbar2 = cfg.Dbar ** 2
    dg = cfg.delta_g
    Yfg, Ufg, Upg = d.Yf @ g_star, d.Uf @ g_star, d.Up @ g_star

    sl_v = slice(0, n_g)
    i_A0 = n_g

    def neg_obj(z):
        v = z[sl_v]; A0 = z[i_A0]
        track = Yfg + dg * (d.Yf @ v) - d.y_ref
        ueff = u + A0                                   # 1_{n3} A0 added elementwise
        return -(track @ track + ueff @ ueff)

    def c_unit(z):                                      # ||v|| <= 1
        v = z[sl_v]
        return 1.0 - v @ v

    def c_Dbound(z):                                    # ||D_tilde|| <= Dbar (Db=0)
        v = z[sl_v]; A0 = z[i_A0]
        Da = (d.u_ini - Upg) + A0 - dg * (d.Up @ v)    # = u_ini + 1 A0 - Up g_tilde
        Dc = (u + A0) - (Ufg + dg * (d.Uf @ v))        # = A0 - dg Uf v
        return Dbar2 - (Da @ Da + Dc @ Dc)

    # warm start: v along grad of tracking term; A0 = 0
    grad_dir = d.Yf.T @ (Yfg - d.y_ref)
    v0 = grad_dir / (np.linalg.norm(grad_dir) + 1e-12)
    z0 = np.concatenate([v0, [0.0]])
    res = minimize(neg_obj, z0, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": c_unit},
                                {"type": "ineq", "fun": c_Dbound}],
                   options={"maxiter": 1000, "ftol": 1e-14})
    v = res.x[sl_v]; A0 = float(res.x[i_A0])
    g_t = g_star + dg * v
    u_eff = u + A0                                      # attacked future input
    Ar = d.Yp @ g_t - d.y_ini                           # deception attack on y_ini (Db=0)
    return {"g_tilde": g_t, "y_pred": d.Yf @ g_t, "u_eff": u_eff,
            "A0": A0, "A_r": Ar, "value": -res.fun, "ok": res.success,
            "stealth_gap": np.linalg.norm(g_star - g_t),
            "y_pred_track_err": np.linalg.norm(d.Yf @ g_t - d.y_ref)}


if __name__ == "__main__":
    from deepc import FIG1, FIG2, build
    from tracking import nominal_tracking
    for cfg in (FIG1, FIG2):
        d = build(cfg)
        nom = nominal_tracking(d)
        atk = attacker_response(d, nom["g_star"])
        def_ = defender_security(d)
        print(f"\n=== {cfg.name} (Dbar={cfg.Dbar}, ||D||={cfg.Dnorm}) ===")
        print(f"  gray  (5) track_err={np.linalg.norm(nom['y_pred']-d.y_ref):.4f}")
        print(f"  red   (8) track_err={atk['y_pred_track_err']:.4f}  "
              f"A0(actuation)={atk['A0']:.3e}  ||A_r||(decep)={np.linalg.norm(atk['A_r']):.3e}  "
              f"stealth_gap={atk['stealth_gap']:.2e} (<=delta_g={cfg.delta_g})")
        print(f"  blue  (9) track_err={def_['y_pred_track_err']:.4f}  ok={def_['ok']}  "
              f"worstA0={def_['A0']:.3e}")
