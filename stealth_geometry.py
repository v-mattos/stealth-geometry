"""Geometric characterization of the stealthy-attack set (extension of Bhowmik et al.,
L-CSS 2025, double-integrator example). NEW module -- does NOT touch the equilibrium
reproduction (deepc/tracking/game/plots); it only reuses the matrices they build.

Central identity (stealthiness). With b=[u_ini;y_ini;u], Hbar=[Up;Yp;Uf], FDbar=I_{n_d},
FAbar=[Fa1;Fa2;Fa3], the true (attack-free) behaviour is b = Hbar g + D, ||D||<=Dbar, and
an attack A that keeps the corrupted data b+FAbar A explainable by (g~,D~) satisfies

        FAbar A = Hbar (g~ - g) + (D~ - D).                                   (identity)

Stealth margin. Write D~ = D + FAbar A (g~ = g); the R:=range(FAbar) part of D is
re-explainable by A, the R^perp part is not, so ||FAbar A + D||^2 = ||FAbar(A - A_c)||^2 +
dist(D,R)^2 with
        A_c    = -pinv(FAbar) D           (= -pinv(FAbar) (P D); cancels the R-part of D)
        rho(D) = sqrt(Dbar^2 - dist(D,R)^2),   dist(D,R) = ||(I - P) D||.
For the example injectors (Fa1=1_{n1} e1^T, Fa2=[0 I_{n2}], Fa3=1_{n3} e1^T) this closes
in form (D = (D1,D2,D3) in the u_ini/y_ini/u blocks; the sensor block D2 does NOT enter,
since the whole middle block lies in R):
        dist(D,R)^2 = ||D1||^2 + ||D3||^2 - (1^T D1 + 1^T D3)^2 / (n1 + n3),
        sigma_min(FAbar) = 1,   sigma_max(FAbar) = sqrt(n1 + n3).

Stealthy set is a BOUNDED ELLIPSOID. Expanding ||FAbar A + D||^2 <= Dbar^2 blockwise gives
        (n1+n3)(A0 - A0_c)^2 + ||A_r - A_r_c||^2 <= rho(D)^2,                  (Prop. 1)
i.e. exactly the ellipsoid ||FAbar(A - A_c)|| <= rho(D), with
        A0_c  = -(1^T D1 + 1^T D3)/(n1+n3),   A_r_c = -D2   (so A_c = -pinv(FAbar) D).
Because FAbar has full column rank (sigma_min = 1), this is BOUNDED in every coordinate:
semi-axes rho(D)/sqrt(n1+n3) along A0 and rho(D) along each of the n2 deception axes A_r.
The deception A_r is NOT free -- it is bounded by ||A_r - A_r_c|| <= rho(D). What IS true is
that the sensor DISTURBANCE D2 does not affect the margin rho(D) (it only shifts the centre
to A_r_c = -D2; dist(D,R) is D2-independent, Lemma 2). The scalar-A0 condition
        sqrt(n1 + n3) * |A0 - A0_c(D)| <= rho(D)                              (shadow)
is the PROJECTION of the ellipsoid onto the A0 axis: a NECESSARY (not sufficient) condition,
obtained by minimising over A_r. Prop. 2 / the impact eq. (11) use only max|A0 - A0_c| =
rho/sqrt(n1+n3) and are unaffected. (Aside: restoring g~ != g widens the set only on the
actuation rows, ||[Up;Uf]|| * delta_g ~ 8e-4 << Dbar; the large ||[Up;Yp;Uf]|| ~ 101 lives
in the sensor rows Yp, absorbed by A_r.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import orth

from deepc import DeePCData


@dataclass
class StealthGeometry:
    """Stealthy-attack geometry for one DeePCData instance. Build via from_data()."""
    FAbar: np.ndarray          # [Fa1;Fa2;Fa3], (n_d x n_a), full column rank
    P: np.ndarray              # orthogonal projector onto R = range(FAbar), (n_d x n_d)
    n1: int
    n2: int
    n3: int
    Dbar: float

    @classmethod
    def from_data(cls, d: DeePCData) -> "StealthGeometry":
        FAbar = d.F_A
        Q = orth(FAbar)                       # orthonormal basis of range(FAbar) (SVD-based)
        P = Q @ Q.T                           # projector; avoids forming (F^T F)^{-1} explicitly
        cfg = d.cfg
        return cls(FAbar, P, cfg.n1, cfg.n2, cfg.n3, cfg.Dbar)

    # --- blocks ---
    def _split(self, D: np.ndarray):
        """Return (D1, D2, D3) = (u_ini, y_ini, u) blocks of a disturbance D in R^{n_d}."""
        n1, n2 = self.n1, self.n2
        return D[:n1], D[n1:n1 + n2], D[n1 + n2:]

    # --- core quantities ---
    def dist_to_R(self, D: np.ndarray) -> float:
        """dist(D, R) = ||(I - P) D||, distance from D to range(FAbar)."""
        return float(np.linalg.norm(D - self.P @ D))

    def dist_to_R_closed_form(self, D: np.ndarray) -> float:
        """Example-specific closed form: sqrt(||D1||^2+||D3||^2-(1^T D1+1^T D3)^2/(n1+n3))."""
        D1, _, D3 = self._split(D)
        s = float(np.sum(D1) + np.sum(D3))
        val = D1 @ D1 + D3 @ D3 - s * s / (self.n1 + self.n3)
        return float(np.sqrt(max(val, 0.0)))

    def rho(self, D: np.ndarray) -> float:
        """Stealth-margin radius rho(D) = sqrt(Dbar^2 - dist(D,R)^2) (0 if D exhausts budget)."""
        dist = self.dist_to_R(D)
        return float(np.sqrt(max(self.Dbar ** 2 - dist ** 2, 0.0)))

    def A_c(self, D: np.ndarray) -> np.ndarray:
        """Recentering attack A_c = -pinv(FAbar) D (cancels the R-component P D of D)."""
        return -np.linalg.pinv(self.FAbar) @ D

    def A0_center(self, D: np.ndarray) -> float:
        """A0_c(D) = e1^T A_c(D), the actuation coordinate of the recentering. For the
        example this is -(1^T D1 + 1^T D3)/(n1+n3)."""
        return float(self.A_c(D)[0])

    def max_marginal_actuation(self, D: np.ndarray) -> float:
        """Largest stealthy |A0 - A0_c(D)|: rho(D)/sqrt(n1+n3) (the A0-semi-axis of the
        ellipsoid, attained when all budget goes to A0, i.e. A_r = A_r_c). This is the
        attacker's actuation authority beyond merely mimicking the disturbance D."""
        return self.rho(D) / np.sqrt(self.n1 + self.n3)

    # --- membership: the FULL ellipsoid (Prop. 1) vs its A0-shadow (necessary only) ---
    def membership_ellipsoid(self, A: np.ndarray, D: np.ndarray):
        """Exact stealthy-set test (Prop. 1). Returns (lhs, slack) with
        lhs = ||FAbar(A - A_c)||^2 = (n1+n3)(A0-A0_c)^2 + ||A_r - A_r_c||^2 and
        slack = rho(D)^2 - lhs (in squared units). Stealthy iff slack >= 0. The deception
        A_r enters through ||A_r - A_r_c||^2 -- it is NOT free."""
        Ac = self.A_c(D)
        lhs = float(np.linalg.norm(self.FAbar @ (A - Ac)) ** 2)
        return lhs, float(self.rho(D) ** 2 - lhs)

    def shadow_check(self, A0: float, D: np.ndarray):
        """A0-shadow of the ellipsoid: the NECESSARY (not sufficient) condition
        sqrt(n1+n3)|A0-A0_c| <= rho(D), obtained by minimising the ellipsoid over A_r.
        Returns (lhs, slack) with lhs = sqrt(n1+n3)|A0-A0_c|, slack = rho(D) - lhs (linear
        units). slack >= 0 is implied by membership but does not imply it."""
        lhs = float(np.sqrt(self.n1 + self.n3) * abs(A0 - self.A0_center(D)))
        return lhs, float(self.rho(D) - lhs)


# ---------------------------------------------------------------------------
# Task 1.3 / 1.4 -- self-tests
# ---------------------------------------------------------------------------
def _run_tests(d: DeePCData, n_random: int = 200, tol: float = 1e-10, seed: int = 0) -> None:
    sg = StealthGeometry.from_data(d)
    rng = np.random.default_rng(seed)
    n_d = sg.FAbar.shape[0]

    # (1.3) closed-form dist vs ||(I-P)D|| for random D
    max_err = 0.0
    for _ in range(n_random):
        D = rng.standard_normal(n_d)
        max_err = max(max_err, abs(sg.dist_to_R(D) - sg.dist_to_R_closed_form(D)))
    assert max_err < tol, f"dist closed-form mismatch: max_err={max_err:.2e} > {tol:.0e}"

    # recentering identity: FAbar A_c = -P D  (so it cancels exactly the R-component of D)
    max_rec = 0.0
    for _ in range(n_random):
        D = rng.standard_normal(n_d)
        max_rec = max(max_rec, np.linalg.norm(sg.FAbar @ sg.A_c(D) + sg.P @ D))
    assert max_rec < tol, f"recentering identity FAbar A_c = -P D failed: {max_rec:.2e}"

    # orthogonal split: ||FAbar A + D||^2 == ||FAbar(A-A_c)||^2 + dist(D,R)^2 (Prop. 1 algebra)
    max_split = 0.0
    for _ in range(50):
        D = rng.standard_normal(n_d)
        A = rng.standard_normal(sg.FAbar.shape[1])
        lhs = np.linalg.norm(sg.FAbar @ A + D) ** 2
        rhs = np.linalg.norm(sg.FAbar @ (A - sg.A_c(D))) ** 2 + sg.dist_to_R(D) ** 2
        max_split = max(max_split, abs(lhs - rhs))
    assert max_split < 1e-8, f"orthogonal-split identity failed: {max_split:.2e}"

    # (1.4) singular values of FAbar: {1 (x n2), sqrt(n1+n3)}
    sv = np.linalg.svd(sg.FAbar, compute_uv=False)
    smin, smax = sv.min(), sv.max()
    exp_max = np.sqrt(sg.n1 + sg.n3)
    assert abs(smin - 1.0) < 1e-9, f"sigma_min(FAbar)={smin:.6f} != 1"
    assert abs(smax - exp_max) < 1e-9, f"sigma_max(FAbar)={smax:.6f} != sqrt(n1+n3)={exp_max:.6f}"

    # (T1) the deception A_r is NOT free: for D=0, A0=0, a large random A_r must be REJECTED
    # by the ellipsoid (slack<0) yet ACCEPTED by the shadow (which ignores A_r).
    n_a = sg.FAbar.shape[1]
    D0 = np.zeros(n_d)
    A_bigr = np.zeros(n_a)
    A_bigr[1:] = 5.0 * sg.Dbar * rng.standard_normal(n_a - 1)   # A0=0, ||A_r|| >> rho=Dbar
    _, slack_ell = sg.membership_ellipsoid(A_bigr, D0)
    _, slack_shadow = sg.shadow_check(A_bigr[0], D0)
    assert slack_ell < 0, f"ellipsoid wrongly accepts a large A_r (slack={slack_ell:.2e})"
    assert slack_shadow >= 0, f"shadow should ignore A_r (slack={slack_shadow:.2e})"
    # a small attack strictly inside the ellipsoid must be accepted by BOTH
    A_small = np.zeros(n_a); A_small[0] = 0.3 * sg.max_marginal_actuation(D0)
    A_small[1:] = 0.1 * sg.Dbar * rng.standard_normal(n_a - 1)
    _, s_ell = sg.membership_ellipsoid(A_small, D0)
    _, s_sh = sg.shadow_check(A_small[0], D0)
    assert s_ell > 0 and s_sh > 0, f"inside-attack rejected (ell={s_ell:.2e}, shadow={s_sh:.2e})"

    print(f"{d.cfg.name}: ALL TESTS PASS | dist closed-form err={max_err:.1e}, "
          f"recentering={max_rec:.1e}, split={max_split:.1e}; "
          f"sigma(FAbar) in [{smin:.4f}, {smax:.4f}] (expect [1, {exp_max:.4f}]); "
          f"ellipsoid rejects large A_r (slack={slack_ell:.1e}<0) while shadow accepts it")


if __name__ == "__main__":
    from deepc import FIG1, FIG2, build
    for cfg in (FIG1, FIG2):
        _run_tests(build(cfg))
