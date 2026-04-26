"""
Experiment A — Step Function H(x)
==================================
Demonstrates SRFL field evolution, defect generation,
swarm lifecycle, and action monitoring on the Heaviside step.

Usage
-----
    python experiments/run_step.py
    python experiments/run_step.py --n 512 --steps 70 --verbose
"""

import argparse
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Allow running from repo root or experiments/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srfl import (
    SRFLField, Swarm, ActionFunctional,
    ScaleProjection, DefectAlgebra
)

# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SRFL Experiment A: Step Function")
    p.add_argument("--n",       type=int,   default=512,  help="Grid size N")
    p.add_argument("--steps",   type=int,   default=70,   help="Scale steps S")
    p.add_argument("--lam0",    type=float, default=1.0,  help="Coarsest scale λ₀")
    p.add_argument("--lam1",    type=float, default=0.015,help="Finest scale λ_S")
    p.add_argument("--dt",      type=float, default=0.28, help="Pseudo-time step Δs")
    p.add_argument("--n_agents",type=int,   default=14,   help="Initial agents")
    p.add_argument("--outdir",  type=str,   default="figures",help="Output directory")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def run_step_experiment(args):
    print("=" * 58)
    print("  SRFL · Experiment A: Step Function H(x)")
    print("=" * 58)

    # ── Setup ────────────────────────────────────────────────────────
    x          = np.linspace(-np.pi, np.pi, args.n)
    dx         = float(x[1] - x[0])
    lam_sched  = np.logspace(
        np.log10(args.lam0),
        np.log10(args.lam1),
        args.steps)
    target     = np.where(x >= 0, 1.0, 0.0).astype(float)

    print(f"  Grid:   N={args.n},  dx={dx:.5f}")
    print(f"  Scale:  λ ∈ [{lam_sched[-1]:.4f}, {lam_sched[0]:.3f}],  S={args.steps}")

    # ── Field evolution ───────────────────────────────────────────────
    print("\n  [1/4] Running field evolution …")
    engine = SRFLField(x, target, lam_sched, dt=args.dt)
    fields, errors = engine.run(verbose=args.verbose)
    print(f"        L²err: {errors[0]:.4f} → {errors[-1]:.4f}")
    rate = engine.convergence_rate(errors)
    print(f"        Convergence rate r ≈ {rate:.3f}  (error ~ λ^r)")

    # ── Swarm ────────────────────────────────────────────────────────
    print("\n  [2/4] Running swarm …")
    swarm = Swarm(x, n_init=args.n_agents)
    for k, (phi, lam) in enumerate(zip(fields[1:], lam_sched[1:]), 1):
        swarm.step(phi, target, lam, k)

    summary = swarm.event_summary()
    print(f"        Events: spawn={summary['spawn']}, "
          f"merge={summary['merge']}, annihilate={summary['annihilate']}")

    # ── Action functional ────────────────────────────────────────────
    print("\n  [3/4] Computing action 𝒜 …")
    action = ActionFunctional(x, lam_sched, target)
    A = action.total(fields)
    print(f"        𝒜_data  = {A['data']:.6f}")
    print(f"        𝒜_scale = {A['scale']:.6f}")
    print(f"        𝒜_total = {A['total']:.6f}")

    # ── Defect detection ─────────────────────────────────────────────
    print("\n  [4/4] Detecting defects in final field …")
    alg     = DefectAlgebra(x)
    defects = alg.detect_from_curvature(x, fields[-1], dx)
    print(f"        Detected {len(defects)} defect(s):")
    for d in defects:
        print(f"          {d}")

    # ── Figures ──────────────────────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)
    _plot_results(x, target, fields, errors, lam_sched, swarm, args.outdir)
    print(f"\n  Figures saved to '{args.outdir}/'")
    print("=" * 58)
    return fields, errors, swarm, A


def _plot_results(x, target, fields, errors, lam_sched, swarm, outdir):
    sel  = np.linspace(0, len(fields) - 1, 8, dtype=int)
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: field evolution
    ax = axes[0]
    for k, idx in enumerate(sel):
        c = k / (len(sel) - 1)
        ax.plot(x, fields[idx], color=cmap(0.9 - 0.75 * c),
                lw=1.8, label=f"λ={lam_sched[idx]:.3f}", alpha=0.9)
    ax.plot(x, target, "r--", lw=1.5, alpha=0.6, label="H(x)")
    ax.set_title("Field Evolution Φ(x,λ)")
    ax.set_xlabel("x"); ax.set_ylabel("Φ")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.2)

    # Panel 2: L² error convergence
    ax = axes[1]
    ax.semilogy(lam_sched, errors, "b-o", ms=3, lw=2)
    ax.invert_xaxis()
    ax.set_xlabel("λ  (→ fine)"); ax.set_ylabel("L² error")
    ax.set_title("Convergence: ‖Φ(·,λ) − H‖_{L²}")
    ax.grid(True, which="both", alpha=0.25)

    # Panel 3: swarm positions over steps
    ax = axes[2]
    n_hist = min(len(swarm.history), len(lam_sched))
    for s, pos in enumerate(swarm.history[:n_hist]):
        c = s / n_hist
        ax.scatter(pos, np.full(len(pos), s), s=12,
                   color=cmap(1 - c), alpha=0.65)
    for (s, etype, ex) in swarm.events:
        if s >= n_hist: continue
        marker = {"spawn": "^", "merge": "D", "annihilate": "x"}[etype]
        color  = {"spawn": "lime", "merge": "gold", "annihilate": "red"}[etype]
        ax.scatter(ex, s, marker=marker, s=80, color=color, zorder=5)
    ax.set_xlabel("x"); ax.set_ylabel("Scale step s")
    ax.set_title("Swarm Dynamics")
    ax.grid(True, alpha=0.2)

    plt.suptitle("SRFL — Step Function H(x)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{outdir}/step_experiment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_step_experiment(parse_args())
