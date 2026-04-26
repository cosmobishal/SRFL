"""
Experiment B — Oscillatory Function x·sin(1/x)
===============================================
Demonstrates SRFL defect generation on a target with an essential
singularity at x=0 requiring infinitely many oscillatory defects.

Usage
-----
    python experiments/run_oscillatory.py
    python experiments/run_oscillatory.py --n 512 --steps 70
"""

import argparse
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srfl import (
    SRFLField, Swarm, ActionFunctional,
    ScaleProjection, DefectAlgebra, OscillatoryDefect
)

# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SRFL Experiment B: Oscillatory Function x·sin(1/x)")
    p.add_argument("--n",       type=int,   default=512)
    p.add_argument("--steps",   type=int,   default=70)
    p.add_argument("--lam0",    type=float, default=1.0)
    p.add_argument("--lam1",    type=float, default=0.015)
    p.add_argument("--dt",      type=float, default=0.28)
    p.add_argument("--n_agents",type=int,   default=14)
    p.add_argument("--outdir",  type=str,   default="figures")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def make_osc_target(x: np.ndarray) -> np.ndarray:
    """f(x) = x·sin(1/x),  f(0) = 0."""
    out = np.zeros_like(x, dtype=float)
    nz  = x != 0
    out[nz] = x[nz] * np.sin(1.0 / x[nz])
    return out


def run_oscillatory_experiment(args):
    print("=" * 58)
    print("  SRFL · Experiment B: x·sin(1/x)")
    print("=" * 58)

    x         = np.linspace(-np.pi, np.pi, args.n)
    dx        = float(x[1] - x[0])
    lam_sched = np.logspace(
        np.log10(args.lam0), np.log10(args.lam1), args.steps)
    target    = make_osc_target(x)

    print(f"  Grid:  N={args.n},  dx={dx:.5f}")
    print(f"  Scale: λ ∈ [{lam_sched[-1]:.4f}, {lam_sched[0]:.3f}]")

    # ── Field evolution ────────────────────────────────────────────
    print("\n  [1/4] Field evolution …")
    engine = SRFLField(x, target, lam_sched, dt=args.dt)
    fields, errors = engine.run(verbose=args.verbose)
    print(f"        L²err: {errors[0]:.4f} → {errors[-1]:.4f}")

    # ── Swarm ──────────────────────────────────────────────────────
    print("\n  [2/4] Swarm …")
    swarm = Swarm(x, n_init=args.n_agents)
    for k, (phi, lam) in enumerate(zip(fields[1:], lam_sched[1:]), 1):
        swarm.step(phi, target, lam, k)
    summary = swarm.event_summary()
    print(f"        Events: {summary}")

    # ── Action ────────────────────────────────────────────────────
    print("\n  [3/4] Action functional …")
    action = ActionFunctional(x, lam_sched, target)
    A = action.total(fields)
    print(f"        𝒜_data={A['data']:.4f}  𝒜_scale={A['scale']:.4f}"
          f"  𝒜_total={A['total']:.4f}")

    # ── Oscillatory defect analysis ────────────────────────────────
    print("\n  [4/4] Oscillatory defect analysis near x=0 …")
    alg     = DefectAlgebra(x)
    defects = alg.detect_from_curvature(x, fields[-1], dx, kappa=0.8, eps_osc=0.5)
    osc_defects = [d for d in defects if isinstance(d, OscillatoryDefect)]
    print(f"        Total defects: {len(defects)}, oscillatory: {len(osc_defects)}")
    for d in osc_defects:
        print(f"          {d}")

    # ── Scale projection consistency ───────────────────────────────
    proj    = ScaleProjection(x)
    sp_err  = proj.consistency_profile(fields, lam_sched, stride=7)
    print(f"\n        Scale-consistency err (mean): {np.mean(sp_err[7:]):.5f}")

    # ── Figures ───────────────────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)
    _plot_results(x, target, fields, errors, lam_sched, swarm,
                  sp_err, args.outdir)
    print(f"\n  Figures → '{args.outdir}/'")
    print("=" * 58)
    return fields, errors, swarm, A


def _plot_results(x, target, fields, errors, lam_sched,
                  swarm, sp_err, outdir):
    sel  = np.linspace(0, len(fields) - 1, 8, dtype=int)
    cmap = plt.cm.plasma

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (0,0) Field evolution
    ax = axes[0, 0]
    for k, idx in enumerate(sel):
        c = k / (len(sel) - 1)
        ax.plot(x, fields[idx], color=cmap(0.9 - 0.75 * c), lw=1.6, alpha=0.9)
    ax.plot(x, target, "r--", lw=1.5, alpha=0.6, label="x·sin(1/x)")
    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.5, 1.5)
    ax.set_title("Field Evolution Φ(x,λ)")
    ax.set_xlabel("x"); ax.set_ylabel("Φ")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

    # (0,1) Zoom near x=0
    ax = axes[0, 1]
    for k, idx in enumerate(sel):
        c = k / (len(sel) - 1)
        ax.plot(x, fields[idx], color=cmap(0.9 - 0.75 * c), lw=1.6, alpha=0.9)
    ax.plot(x, target, "r--", lw=1.5, alpha=0.6)
    ax.set_xlim(-0.4, 0.4); ax.set_ylim(-0.35, 0.35)
    ax.set_title("Zoom: Oscillatory Singularity at x=0")
    ax.set_xlabel("x"); ax.set_ylabel("Φ")
    ax.grid(True, alpha=0.2)

    # (1,0) L² error
    ax = axes[1, 0]
    ax.semilogy(lam_sched, errors, "b-o", ms=3, lw=2)
    ax.invert_xaxis()
    ax.set_xlabel("λ"); ax.set_ylabel("L² error")
    ax.set_title("Convergence ‖Φ − f‖_{L²}")
    ax.grid(True, which="both", alpha=0.25)

    # (1,1) Scale consistency
    ax = axes[1, 1]
    steps = np.arange(len(sp_err))
    ax.semilogy(steps[7:], sp_err[7:] + 1e-12, "g-", lw=2)
    ax.set_xlabel("Scale step s"); ax.set_ylabel("Consistency error")
    ax.set_title("Scale Consistency ‖Φ(λₖ) − Π·Φ(λₖ₋₁)‖")
    ax.grid(True, which="both", alpha=0.25)

    plt.suptitle("SRFL — Oscillatory Function x·sin(1/x)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{outdir}/oscillatory_experiment.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_oscillatory_experiment(parse_args())
