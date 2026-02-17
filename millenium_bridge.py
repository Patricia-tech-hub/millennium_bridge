"""
Millennium Bridge + Pedestrians (Strogatz et al.) coupled simulation
with controlled "crowd ramp" experiment.

Model:
  M X¨ + B X˙ + K X = sum_{i=1..N} G sin(theta_i)
  theta˙_i = Omega_i + C A(t) sin(Psi(t) - theta_i + alpha)

Bridge amplitude/phase (instantaneous):
  A = sqrt( X^2 + (X˙/Omega)^2 )
  Psi = atan2( X, X˙/Omega )

Order parameter:
  R e^{iPhi} = (1/N) sum_j e^{i theta_j}

Experiment:
  Start N0 pedestrians, then every ΔT seconds add ΔN pedestrians (new theta, new Omega_i)
  until Nmax, then run a bit longer to see post-threshold dynamics.

Outputs:
  - Computes theoretical Nc for Normal freq distribution (mean Omega, std sigma)
  - Computes tc = first time N(t) >= Nc
  - Produces 3 stacked plots: N(t), A(t), R(t) with a vertical dashed line at t=tc ("N = Nc")

Defaults are "sensible" and can be changed via CLI args.
"""

from __future__ import annotations

import math
import argparse
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ----------------------------
# Parameters
# ----------------------------

@dataclass(frozen=True)
class Params:
    M: float = 1.13e5            # kg
    B: float = 1.1e4            # kg/s
    K: float = 4.73e6           # kg/s^2  (N/m)
    G: float = 30.0             # N  (kg*m/s^2)
    C: float = 16.0             # m^-1 s^-1
    alpha: float = 0.5 * math.pi  # rad (note: equivalent to 0 mod 2π)
    Omega: float = 6.47         # rad/s (bridge natural frequency)
    sigma: float = 0.63         # rad/s (std dev of pedestrian frequencies)


# ----------------------------
# Math helpers
# ----------------------------

def bridge_amp_phase(X: np.ndarray, V: np.ndarray, Omega: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute A(t), Psi(t) from X(t), V(t)=Xdot(t).
    Psi uses atan2(X, V/Omega) consistent with:
      X = A sin(Psi), V = A Omega cos(Psi)
    """
    V_over = V / Omega
    A = np.sqrt(X * X + V_over * V_over)
    Psi = np.arctan2(X, V_over)
    return A, Psi


def kuramoto_order_parameter(theta: np.ndarray) -> Tuple[float, float]:
    """
    Given theta array shape (N,), compute (R, Phi) where:
      R e^{iPhi} = (1/N) sum exp(i theta_j)
    """
    z = np.mean(np.exp(1j * theta))
    return float(np.abs(z)), float(np.angle(z))


def theoretical_Nc(B: float, Omega: float, G: float, C: float, sigma: float) -> float:
    """
    Using:
      Nc = (2 B Omega) / (pi G C P(Omega))
    and for Normal(mean=Omega, std=sigma):
      P(Omega) = 1/(sqrt(2*pi)*sigma)
    so:
      Nc = (2 B Omega) * (sqrt(2*pi)*sigma) / (pi G C)
    """
    P_at_mean = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)
    Nc = (2.0 * B * Omega) / (math.pi * G * C * P_at_mean)
    return Nc


def damping_ratio(B: float, M: float, Omega: float) -> float:
    """For a 2nd-order oscillator: zeta = B / (2 M Omega)."""
    return B / (2.0 * M * Omega)


# ----------------------------
# ODE RHS factory (fixed N on each interval)
# ----------------------------

def make_rhs(params: Params, omega_i: np.ndarray):
    """
    Return rhs(t, y) for fixed omega_i (length N) on a given interval.

    State vector y:
      y[0] = X
      y[1] = V = Xdot
      y[2:] = theta (length N)
    """
    M, B, K, G, C, alpha, Omega = (
        params.M, params.B, params.K, params.G, params.C, params.alpha, params.Omega
    )

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        X = y[0]
        V = y[1]
        theta = y[2:]

        # Bridge amplitude/phase at current instant
        A = math.sqrt(X * X + (V / Omega) ** 2)
        Psi = math.atan2(X, V / Omega)

        # Bridge acceleration
        forcing = G * np.sum(np.sin(theta))
        Xdd = (forcing - B * V - K * X) / M

        # Pedestrian phases
        dtheta = omega_i + C * A * np.sin(Psi - theta + alpha)

        out = np.empty_like(y)
        out[0] = V
        out[1] = Xdd
        out[2:] = dtheta
        return out

    return rhs


# ----------------------------
# Simulation
# ----------------------------

@dataclass
class SimConfig:
    N0: int = 50
    dN: int = 10
    Nmax: int = 250
    dT: float = 120.0          # seconds between crowd increments
    dt_eval: float = 0.02     # time spacing for t_eval output (smooth plots)
    extra_intervals_after_max: int = 20  # run extra dT intervals after reaching Nmax
    seed: int = 123
    X0: float = 0.0
    V0: float = 0.0
    X0_jitter: float = 1e-6   # small initial bridge displacement to break symmetry (can be 0)


def run_ramp_experiment(params: Params, cfg: SimConfig):
    rng = np.random.default_rng(cfg.seed)

    # Initial crowd
    N = cfg.N0
    theta = rng.uniform(0.0, 2.0 * math.pi, size=N)
    omega_i = rng.normal(loc=params.Omega, scale=params.sigma, size=N)

    # Initial bridge conditions (small jitter helps avoid perfectly symmetric forcing = 0)
    X0 = cfg.X0 + rng.normal(scale=cfg.X0_jitter)
    V0 = cfg.V0
    y0 = np.concatenate(([X0, V0], theta))

    # Determine number of ramp steps until Nmax
    if cfg.Nmax < cfg.N0:
        raise ValueError("Nmax must be >= N0")

    steps_to_max = int(math.ceil((cfg.Nmax - cfg.N0) / cfg.dN)) if cfg.Nmax > cfg.N0 else 0
    total_intervals = steps_to_max + cfg.extra_intervals_after_max

    # Storage arrays
    t_all: List[float] = []
    X_all: List[float] = []
    V_all: List[float] = []
    A_all: List[float] = []
    R_all: List[float] = []
    N_all: List[int] = []

    t0 = 0.0

    for k in range(total_intervals):
        # End time of this interval
        t1 = t0 + cfg.dT

        # Build RHS with current omega_i (fixed on this interval)
        rhs = make_rhs(params, omega_i)

        # Evaluation times: include endpoint to make stair plots align nicely
        # Ensure t_eval does not exceed t1 due to floating-point rounding
        t_eval = np.arange(t0, t1 + cfg.dt_eval * 0.5, cfg.dt_eval)
        t_eval = t_eval[t_eval <= t1]  # Filter out any values exceeding t1

        sol = solve_ivp(
            rhs,
            t_span=(t0, t1),
            y0=y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
            vectorized=False,
        )

        if not sol.success:
            raise RuntimeError(f"ODE solve failed on interval {k}: {sol.message}")

        # Unpack results
        t = sol.t
        X = sol.y[0]
        V = sol.y[1]
        theta_t = sol.y[2:]  # shape (N, len(t))

        # Compute A(t), Psi(t) and R(t)
        A, _Psi = bridge_amp_phase(X, V, params.Omega)
        # R(t) for each time sample
        # Compute mean(exp(i theta)) across pedestrians for each column
        z = np.mean(np.exp(1j * theta_t), axis=0)
        R = np.abs(z)

        # Append to global storage (avoid duplicating first point except at start)
        start_idx = 0 if len(t_all) == 0 else 1

        t_all.extend(t[start_idx:].tolist())
        X_all.extend(X[start_idx:].tolist())
        V_all.extend(V[start_idx:].tolist())
        A_all.extend(A[start_idx:].tolist())
        R_all.extend(R[start_idx:].tolist())
        N_all.extend([N] * int(len(t[start_idx:])))

        # Prepare for next interval: take final state
        y_end = sol.y[:, -1]
        y0 = y_end
        t0 = t1

        # Ramp: add pedestrians at end of interval if we haven't reached Nmax yet
        if N < cfg.Nmax:
            add = min(cfg.dN, cfg.Nmax - N)
            if add > 0:
                new_theta = rng.uniform(0.0, 2.0 * math.pi, size=add)
                new_omega = rng.normal(loc=params.Omega, scale=params.sigma, size=add)

                # Extend theta state in y0 and omega_i
                y0 = np.concatenate((y0[:2], y0[2:], new_theta))
                omega_i = np.concatenate((omega_i, new_omega))
                N += add

    # Convert to arrays
    t_all = np.asarray(t_all)
    X_all = np.asarray(X_all)
    V_all = np.asarray(V_all)
    A_all = np.asarray(A_all)
    R_all = np.asarray(R_all)
    N_all = np.asarray(N_all)

    return t_all, X_all, V_all, A_all, R_all, N_all


# ----------------------------
# Plotting + summary
# ----------------------------

def plot_results(t, A, R, N, Nc, tc, title="Millennium Bridge crowd-ramp experiment"):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 8))

    # N(t)
    axes[0].plot(t, N)
    axes[0].set_ylabel("N(t)")
    axes[0].set_title(title)

    # A(t)
    axes[1].plot(t, A)
    axes[1].set_ylabel("A(t) [m]")

    # R(t)
    axes[2].plot(t, R)
    axes[2].set_ylabel("R(t)")
    axes[2].set_xlabel("time t [s]")
    axes[2].set_ylim(-0.02, 1.02)

    # Vertical line at tc if it exists
    if np.isfinite(tc):
        for ax in axes:
            ax.axvline(tc, linestyle="--")
        # Put label near top of first axis
        axes[0].text(tc, axes[0].get_ylim()[1], "  N = Nc", va="top")

    # Annotate Nc
    axes[0].axhline(Nc, linestyle=":", linewidth=1)
    axes[0].text(t[0], Nc, f" Nc ≈ {Nc:.1f}", va="bottom")

    fig.tight_layout()
    plt.show()


def summarize_onset(t, A, R, N, Nc, tc):
    # Simple onset heuristics: when A exceeds a small threshold and stays there briefly
    A_thr = max(1e-4, 5.0 * np.median(A[: max(10, len(A)//20)]))  # adaptive-ish
    idx = np.where(A > A_thr)[0]
    t_onset_A = t[idx[0]] if len(idx) else float("nan")

    R_thr = 0.2
    idxR = np.where(R > R_thr)[0]
    t_onset_R = t[idxR[0]] if len(idxR) else float("nan")

    print("\n--- Theoretical vs observed onset ---")
    print(f"Nc (theory) ≈ {Nc:.3f}")
    if np.isfinite(tc):
        print(f"tc (first time N(t) >= Nc) ≈ {tc:.3f} s")
    else:
        print("tc: N(t) never reaches Nc in this run.")

    print("\nObserved (heuristic) onset times:")
    print(f"  First A(t) > {A_thr:.2e}  at t ≈ {t_onset_A:.3f} s")
    print(f"  First R(t) > {R_thr:.2f}       at t ≈ {t_onset_R:.3f} s")

    print("\nQualitative expectation:")
    print("  Before tc: A(t) stays small; R(t) fluctuates near ~O(N^{-1/2}).")
    print("  After  tc: feedback can lock phases, R(t) grows to O(1) and A(t) rises sharply.")


# ----------------------------
# CLI / main
# ----------------------------

def main():
    p = argparse.ArgumentParser(description="Millennium Bridge crowd ramp simulation")
    p.add_argument("--N0", type=int, default=50)
    p.add_argument("--dN", type=int, default=10)
    p.add_argument("--Nmax", type=int, default=250)
    p.add_argument("--dT", type=float, default=5.0)
    p.add_argument("--dt", type=float, default=0.02, help="t_eval spacing for smooth plots")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--extra", type=int, default=3, help="extra ΔT intervals after reaching Nmax")
    args = p.parse_args()

    params = Params()

    # Check / report derived quantities
    Omega_from_MK = math.sqrt(params.K / params.M)
    zeta = damping_ratio(params.B, params.M, params.Omega)
    Nc = theoretical_Nc(params.B, params.Omega, params.G, params.C, params.sigma)

    print("--- Parameters ---")
    print(f"M={params.M:.3e} kg, B={params.B:.3e} kg/s, K={params.K:.3e} kg/s^2")
    print(f"G={params.G:.3f} N, C={params.C:.3f} m^-1 s^-1, alpha={params.alpha:.3f} rad")
    print(f"Omega (given)={params.Omega:.4f} rad/s, Omega (sqrt(K/M))={Omega_from_MK:.4f} rad/s")
    print(f"sigma={params.sigma:.4f} rad/s")
    print(f"zeta (from B/(2 M Omega))={zeta:.4f}")
    print(f"Nc (theory, Normal P) ≈ {Nc:.4f}")

    # Compute tc from staircase schedule
    # N(t) increases by dN every dT until Nmax
    N0, dN, Nmax, dT = args.N0, args.dN, args.Nmax, args.dT
    if Nc <= N0:
        k_star = 0
        tc = 0.0
    elif Nc > Nmax:
        tc = float("inf")
    else:
        k_star = int(math.ceil((Nc - N0) / dN))
        tc = k_star * dT

    print(f"tc (schedule-based) ≈ {tc if np.isfinite(tc) else 'inf'}")

    cfg = SimConfig(
        N0=args.N0,
        dN=args.dN,
        Nmax=args.Nmax,
        dT=args.dT,
        dt_eval=args.dt,
        extra_intervals_after_max=args.extra,
        seed=args.seed,
    )

    t, X, V, A, R, N = run_ramp_experiment(params, cfg)

    # Plot + summary
    plot_results(t, A, R, N, Nc=Nc, tc=tc)
    summarize_onset(t, A, R, N, Nc=Nc, tc=tc)


if __name__ == "__main__":
    main()
