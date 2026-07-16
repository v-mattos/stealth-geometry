"""Attack-free robust tracking -- eq. (5). Produces the nominal g* (the GRAY curve).

Problem (5), with the example injection matrices (Fd1=[I 0 0], Fd2=[0 I 0],
Fd3=[0 0 I]) the structure collapses nicely. Write D = [D_a; D_b; D_c] with
D_a in R^n1 (init input), D_b in R^n2 (init output), D_c in R^n3 (future input).
The equality [u_ini; y_ini; u] = [Up;Yp;Uf] g + [Fd1;Fd2;Fd3] D gives

    D_a = u_ini - Up g ,   D_b = y_ini - Yp g ,   u = Uf g + D_c ,

so D_a, D_b are PINNED by g and only D_c is free. The control effort
||u||^2 = ||Uf g + D_c||^2 and the inner maximization is

    max_{||D||<=Dbar} ||Uf g + D_c||^2
      s.t.  ||D_a||^2 + ||D_b||^2 + ||D_c||^2 <= Dbar^2   (D_a,D_b fixed)
    =>  worst ||u|| = ||Uf g|| + r(g),   r(g) = sqrt(Dbar^2 - ||D_a||^2 - ||D_b||^2),

the classic "align the residual budget with Uf g" worst case. Hence (5) becomes the
small smooth program

    min_g  ||Yf g - yref||^2 + (||Uf g|| + r(g))^2 + lambda_g ||g||^2
    s.t.   ||u_ini - Up g||^2 + ||y_ini - Yp g||^2 <= Dbar^2   (so r(g) is real).

This is nonconvex (product/sqrt coupling) but only n_g~24-28 dims and very smooth;
we solve it with SLSQP from the least-squares warm start. The gray curve is Yf g*.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from deepc import DeePCData


def _initial_mismatch_sq(g, d: DeePCData):
    """||D_a||^2 + ||D_b||^2  =  ||u_ini - Up g||^2 + ||y_ini - Yp g||^2."""
    ea = d.u_ini - d.Up @ g
    eb = d.y_ini - d.Yp @ g
    return ea @ ea + eb @ eb


def nominal_tracking(d: DeePCData, warm: np.ndarray | None = None):
    """Solve (5). Returns dict with g_star, y_pred (=Yf g*), u_nom (=Uf g*)."""
    cfg = d.cfg
    Dbar2 = cfg.Dbar ** 2

    def objective(g):
        track = d.Yf @ g - d.y_ref
        nu = np.linalg.norm(d.Uf @ g)
        mism = _initial_mismatch_sq(g, d)
        r2 = max(Dbar2 - mism, 0.0)
        r = np.sqrt(r2)
        return track @ track + (nu + r) ** 2 + cfg.lambda_g * (g @ g)

    # feasibility: initial mismatch within the disturbance budget (so D explains u_ini,y_ini)
    constr = {"type": "ineq",
              "fun": lambda g: Dbar2 - _initial_mismatch_sq(g, d)}

    # warm start: least-squares min_g ||Yf g - yref||^2 + ||Uf g||^2 + lambda_g||g||^2
    if warm is None:
        Aq = d.Yf.T @ d.Yf + d.Uf.T @ d.Uf + cfg.lambda_g * np.eye(cfg.n_g)
        bq = d.Yf.T @ d.y_ref
        warm = np.linalg.solve(Aq, bq)

    res = minimize(objective, warm, method="SLSQP", constraints=[constr],
                   options={"maxiter": 500, "ftol": 1e-12})
    g_star = res.x
    return {
        "g_star": g_star,
        "y_pred": d.Yf @ g_star,
        "u_nom": d.Uf @ g_star,
        "cost": res.fun,
        "ok": res.success,
        "init_mismatch": np.sqrt(_initial_mismatch_sq(g_star, d)),
    }


if __name__ == "__main__":
    from deepc import FIG1, FIG2, build
    for cfg in (FIG1, FIG2):
        d = build(cfg)
        out = nominal_tracking(d)
        yp = out["y_pred"].reshape(-1, 2)
        print(f"{cfg.name}: ok={out['ok']} cost={out['cost']:.4f} "
              f"||g*||={np.linalg.norm(out['g_star']):.3f} "
              f"init_mismatch={out['init_mismatch']:.2e} (<=Dbar={cfg.Dbar}) | "
              f"y_pred final (pos,vel)=({yp[-1,0]:.3f},{yp[-1,1]:.3f}) "
              f"target=({cfg.y_ref_pos},{cfg.y_ref_vel})")
