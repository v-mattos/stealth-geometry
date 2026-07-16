"""DeePC primitives for reproducing Bhowmik, Bopardikar & Hespanha (L-CSS 2025),
"Data-Driven Robust Control under Input-Output Stealthy Attacks".

This module builds everything that is *common* to the three curves the paper plots
(gray / red / blue): the offline data, the Hankel predictor blocks (Up, Uf, Yp, Yf),
the disturbance- and attack-injection matrices, and the (consistent) initial
trajectory. The game itself lives in tracking.py / game.py.

Notation map (paper symbol  ->  code identifier)
------------------------------------------------
  A_x, B, C_y            ->  A_state, B, C_y        (eq. 1; NO feedthrough, C_y = I_2)
  T_ini, T_future, T     ->  Tini, Tf, T            (windows / offline data length)
  m, p, n                ->  m, p, n                 (in/out/state dims)
  U_p, U_f, Y_p, Y_f     ->  Up, Uf, Yp, Yf          (Hankel partition, eq. 2)
  n_g = T-Tini-Tf+1      ->  n_g  (= dim g)          (eq. 4)
  D, ||D|| <= Dbar       ->  D_dist, Dbar            (disturbance vs its bound)
  n_d = (m+p)Tini+mTf    ->  n_d  (= dim D)          (eq. 5 text)
  A (attack), n_a=n2+1   ->  attack vec, n_a         (eq. 7)
  F_di, F_ai             ->  Fd1..Fd3, Fa1..Fa3      (Section III)
  Fbar_D, F_D, etc.      ->  Fbar_D, F_D, Fbar_A,..  (Section IV compact notation)

"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# System (eq. 1): double integrator, position control. m=1, p=2, n=2.
# Keep A_state (= A_x, dynamics) distinct from any attack vector A_atk.
# No feedthrough term: the output map is C_y, there is no D_u u term.
# ---------------------------------------------------------------------------
A_state = np.array([[1.0, 0.5],
                    [0.0, 1.0]])
B = np.array([[0.0],
              [1.0]])
C_y = np.array([[1.0, 0.0],
                [0.0, 1.0]])          # = I_2, both states measured (pos, vel)

m = B.shape[1]                        # inputs  = 1
p = C_y.shape[0]                      # outputs = 2
n = A_state.shape[0]                  # states  = 2


@dataclass
class FigConfig:
    """All parameters for one figure (Fig.1 vs Fig.2 differ; see p.6 captions)."""
    name: str
    Tini: int
    Tf: int
    T: int
    lambda_g: float
    delta_g: float
    Dbar: float           # disturbance BOUND  (= Delta_D in the paper)
    Dnorm: float          # TRUE ||D|| realized online (0 for Fig.1, 0.01 for Fig.2)
    y_ref_pos: float = 0.5
    y_ref_vel: float = 0.0
    seed: int = 0
    init_scale: float = 0.25      # [A3] amplitude of the arbitrary initial-trajectory input
    init_seed: int = 10           # [A3] seed for u_ini (separate from the offline-data seed)

    # derived dimensions (eq. 5 text / caption formulas)
    @property
    def n1(self) -> int: return m * self.Tini          # init-input  disturbance block
    @property
    def n2(self) -> int: return p * self.Tini          # init-output disturbance block
    @property
    def n3(self) -> int: return m * self.Tf            # future-input disturbance block
    @property
    def n_d(self) -> int: return self.n1 + self.n2 + self.n3   # dim D
    @property
    def n_a(self) -> int: return self.n2 + 1                   # dim A   (eq. 7 example)
    @property
    def n_g(self) -> int: return self.T - self.Tini - self.Tf + 1   # dim g (eq. 4)
    @property
    def L(self) -> int: return self.Tini + self.Tf     # Hankel depth

    def check_T(self) -> None:
        """Persistency bound (3): T >= (m+1)(Tini+Tf+n) - 1, T set to the minimum."""
        T_min = (m + 1) * (self.Tini + self.Tf + n) - 1
        assert self.T == T_min, (
            f"{self.name}: T={self.T} but (3) minimum is {T_min}")


# Fig.1: ||D|| = 0 (no disturbance online) ; Fig.2: ||D|| = 0.01 == Dbar (full budget).
# Dbar is FIXED at 0.01 across both (the paper's "for a fixed Dbar, as ||D|| -> Dbar..."
# narrative). NOTE: Fig.1 caption literally prints "D in R^25" but the paper's OWN formula
# n_d=(m+p)Tini+mTf gives 30 (and Fig.2's 42 matches the formula) -> caption is a typo;
# we use n_d = 30 and flag it. See README.
FIG1 = FigConfig(name="Fig.1", Tini=5, Tf=15, T=43,
                 lambda_g=1.0, delta_g=1e-4, Dbar=0.01, Dnorm=0.0, seed=1)
FIG2 = FigConfig(name="Fig.2", Tini=9, Tf=15, T=51,
                 lambda_g=1.0, delta_g=1e-4, Dbar=0.01, Dnorm=0.01, seed=2)


# ---------------------------------------------------------------------------
# Offline data + Hankel blocks
# ---------------------------------------------------------------------------
def simulate(u_seq: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Roll out (1): x_{t+1}=A_x x_t + B u_t, y_t = C_y x_t. Returns y_seq (p, N).
    u_seq has shape (m, N). NOISE-FREE (assumption [A2])."""
    N = u_seq.shape[1]
    y = np.zeros((p, N))
    x = x0.reshape(n, 1).copy()
    for t in range(N):
        y[:, t] = (C_y @ x).ravel()
        x = A_state @ x + B @ u_seq[:, t:t + 1]
    return y


def block_hankel(data: np.ndarray, L: int) -> np.ndarray:
    """Block-Hankel of depth L from data (d, T). Returns (d*L, T-L+1) (eq. 2)."""
    d, T = data.shape
    cols = T - L + 1
    H = np.zeros((d * L, cols))
    for i in range(L):
        H[i * d:(i + 1) * d, :] = data[:, i:i + cols]
    return H


@dataclass
class DeePCData:
    """The offline-built objects shared by all three curves, for one FigConfig."""
    cfg: FigConfig
    Up: np.ndarray
    Uf: np.ndarray
    Yp: np.ndarray
    Yf: np.ndarray
    # injection matrices
    Fd1: np.ndarray; Fd2: np.ndarray; Fd3: np.ndarray
    Fa1: np.ndarray; Fa2: np.ndarray; Fa3: np.ndarray
    # initial trajectory (assumption [A3]) and reference
    u_ini: np.ndarray
    y_ini: np.ndarray
    y_ref: np.ndarray

    # --- stacked compact notation (Section IV) ---
    @property
    def Fbar_D(self): return np.vstack([self.Fd1, self.Fd2])          # F_D bar
    @property
    def F_D(self):    return np.vstack([self.Fd1, self.Fd2, self.Fd3])
    @property
    def Fbar_A(self): return np.vstack([self.Fa1, self.Fa2])          # F_A bar
    @property
    def F_A(self):    return np.vstack([self.Fa1, self.Fa2, self.Fa3])
    @property
    def Hbar(self):   return np.vstack([self.Up, self.Yp])
    @property
    def H(self):      return np.vstack([self.Up, self.Yp, self.Uf])
    @property
    def Q_f(self):    return self.Yf.T @ self.Yf + self.Uf.T @ self.Uf


def build(cfg: FigConfig) -> DeePCData:
    """Generate data, Hankel blocks, injection matrices, and the initial trajectory."""
    cfg.check_T()
    rng = np.random.default_rng(cfg.seed)

    # [A1][A2] persistently-exciting offline input -> noise-free output
    u_d = rng.standard_normal((m, cfg.T))
    x0_d = rng.standard_normal((n, 1))
    y_d = simulate(u_d, x0_d)

    Hu = block_hankel(u_d, cfg.L)
    Hy = block_hankel(y_d, cfg.L)
    Up = Hu[:m * cfg.Tini, :]
    Uf = Hu[m * cfg.Tini:, :]
    Yp = Hy[:p * cfg.Tini, :]
    Yf = Hy[p * cfg.Tini:, :]

    L = cfg.L
    Hin = np.vstack([Up, Uf])
    assert np.linalg.matrix_rank(Hin) == m * L, (
        f"{cfg.name}: input Hankel not full row rank -> u^d not PE; change seed [A1]")
    stacked = np.vstack([Up, Yp, Uf, Yf])
    r = np.linalg.matrix_rank(stacked)
    assert r == m * L + n, (
        f"{cfg.name}: stacked Hankel rank {r} != m*L+n={m * L + n} "
        f"(Willems condition violated)")

    # injection matrices (Section III). n1,n2,n3 blocks of D; attack A in R^{n2+1}.
    n1, n2, n3, n_d = cfg.n1, cfg.n2, cfg.n3, cfg.n_d
    Fd1 = np.hstack([np.eye(n1), np.zeros((n1, n2 + n3))])
    Fd2 = np.hstack([np.zeros((n2, n1)), np.eye(n2), np.zeros((n2, n3))])
    Fd3 = np.hstack([np.zeros((n3, n1 + n2)), np.eye(n3)])

    e1 = np.zeros((n2 + 1, 1)); e1[0, 0] = 1.0
    Fa1 = np.ones((n1, 1)) @ e1.T                      # 1_{n1} e1^T
    Fa2 = np.hstack([np.zeros((n2, 1)), np.eye(n2)])   # [0 I_{n2}]
    Fa3 = np.ones((n3, 1)) @ e1.T                      # 1_{n3} e1^T

    # existence condition (Remark 1 / Thm 2): Fbar_A and F_A FULL COLUMN RANK.
    for nameM, M in (("Fbar_A", np.vstack([Fa1, Fa2])),
                     ("F_A", np.vstack([Fa1, Fa2, Fa3]))):
        rc = np.linalg.matrix_rank(M)
        assert rc == M.shape[1], (
            f"{cfg.name}: {nameM} not full column rank ({rc} < {M.shape[1]})")

    # [A3] initial trajectory: an arbitrary (seeded) input simulated through the TRUE system
    # from rest -> an exact attack-free behavior that enters the horizon with downward velocity
    # (reproduces the paper's dip-below-zero transient). init_scale sets the dip depth.
    rng_init = np.random.default_rng(cfg.init_seed)
    u_ini_seq = cfg.init_scale * rng_init.standard_normal((m, cfg.Tini))
    y_ini_seq = simulate(u_ini_seq, np.zeros(n))
    u_ini = u_ini_seq.T.ravel()
    y_ini = y_ini_seq.T.ravel()

    y_ref = np.tile([cfg.y_ref_pos, cfg.y_ref_vel], cfg.Tf)   # (p*Tf,)

    return DeePCData(cfg, Up, Uf, Yp, Yf, Fd1, Fd2, Fd3, Fa1, Fa2, Fa3,
                     u_ini, y_ini, y_ref)


def calculate_analytical_delta_g(d: DeePCData) -> None:
    """
    Calcula o bound real (delta_g) derivado analiticamente em função 
    do limite de perturbação (Dbar) e da norma do distúrbio (Dnorm).
    """
    # Matrizes compactas do artigo
    Hbar = d.Hbar
    Fbar_D = d.Fbar_D
    Fbar_A = d.Fbar_A
    
    # 1. Calcular a matriz de projeção ortogonal no espaço nulo de Fbar_A^T
    # Matematicamente: P_A_perp = I - Fbar_A * (Fbar_A^T * Fbar_A)^dagger * Fbar_A^T
    # Em numpy, np.linalg.pinv(Fbar_A) já calcula (Fbar_A^T * Fbar_A)^-1 * Fbar_A^T
    I = np.eye(Fbar_A.shape[0])
    P_A_perp = I - (Fbar_A @ np.linalg.pinv(Fbar_A))
    
    # 2. Projetar as matrizes Hbar e Fbar_D no espaço não atacado
    M_H = P_A_perp @ Hbar
    M_D = P_A_perp @ Fbar_D
    
    # 3. Calcular a pseudo-inversa de M_H
    M_H_pinv = np.linalg.pinv(M_H)
    
    # 4. Calcular a norma induzida (norma-2) da matriz resultante
    matrix_to_norm = M_H_pinv @ M_D
    N_f = np.linalg.norm(matrix_to_norm, ord=2)
    
    # 5. Calcular os bounds analíticos
    # Pior caso absoluto: ||D||_2 = Delta_D (Dbar)
    delta_g_worst = 2 * d.cfg.Dbar * N_f
    
    # Caso exato da simulação: (||D||_2 + Delta_D) * N_f
    delta_g_actual = (d.cfg.Dnorm + d.cfg.Dbar) * N_f
    
    print("-" * 60)
    print(f"Análise de Furtividade para {d.cfg.name}:")
    print(f"Norma induzida N_f = ||(P_A^perp Hbar)^dagger P_A^perp Fbar_D||_2: {N_f:.4f}")
    print(f"Delta_D (Dbar) definido: {d.cfg.Dbar}")
    print(f"||D||_2 real na simulação: {d.cfg.Dnorm}")
    print(f" -> delta_g arbitrário do artigo: {d.cfg.delta_g:.4e}")
    print(f" -> delta_g rigoroso (pior caso): {delta_g_worst:.4e}")
    print(f" -> delta_g rigoroso (caso atual): {delta_g_actual:.4e}")
    
    # Avaliação: o delta_g do artigo é excessivamente conservador ou otimista?
    if d.cfg.delta_g > delta_g_actual:
        print("AVISO: O delta_g do artigo é MAIOR que o bound analítico.")
        print("O atacante não conseguiria garantir furtividade com essa margem na prática.")
    else:
        print("OK: O delta_g do artigo é seguro (menor que o limite físico real).")
    print("-" * 60)

if __name__ == "__main__":
    for cfg in (FIG1, FIG2):
        d = build(cfg)
        calculate_analytical_delta_g(d)

# if __name__ == "__main__":
#     for cfg in (FIG1, FIG2):
#         d = build(cfg)
#         print(f"{cfg.name}: n_g={cfg.n_g}  n_d={cfg.n_d}  n_a={cfg.n_a}  "
#               f"Up{d.Up.shape} Uf{d.Uf.shape} Yp{d.Yp.shape} Yf{d.Yf.shape}  "
#               f"predictor full row rank OK, Fbar_A/F_A full col rank OK")
