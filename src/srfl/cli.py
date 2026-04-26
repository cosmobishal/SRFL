"""
srfl.cli
========
Command-line interface for SRFL.

Usage
-----
    srfl-run --target step   --n 512 --steps 70 --outdir figures
    srfl-run --target osc    --n 512 --steps 70 --outdir figures
    srfl-run --target both   --outdir figures --verbose
    srfl-run --list-targets
"""

import argparse
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")

from .field      import SRFLField
from .swarm      import Swarm
from .action     import ActionFunctional
from .multiscale import ScaleProjection
from .defects    import DefectAlgebra


TARGETS = {
    "step": {
        "fn"   : lambda x: np.where(x >= 0, 1.0, 0.0).astype(float),
        "label": "Heaviside step  H(x)",
    },
    "osc": {
        "fn"   : lambda x: np.where(x != 0, x * np.sin(1.0 / np.where(x != 0, x, 1.0)), 0.0),
        "label": "Oscillatory  x·sin(1/x)",
    },
    "sine": {
        "fn"   : lambda x: np.sin(2 * x),
        "label": "Sine  sin(2x)",
    },
    "gaussian": {
        "fn"   : lambda x: np.exp(-2 * x**2),
        "label": "Gaussian  exp(-2x²)",
    },
    "sawtooth": {
        "fn"   : lambda x: (x / np.pi) % 1.0 - 0.5,
        "label": "Sawtooth  x/π mod 1",
    },
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="srfl-run",
        description="Swarm Renormalization Field Learning — CLI runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--target",     default="step",
                   choices=list(TARGETS.keys()) + ["both", "all"],
                   help="Target function to learn")
    p.add_argument("--n",          type=int,   default=512,  help="Grid size N")
    p.add_argument("--steps",      type=int,   default=70,   help="Scale steps S")
    p.add_argument("--lam0",       type=float, default=1.0,  help="Coarsest scale λ₀")
    p.add_argument("--lam1",       type=float, default=0.015,help="Finest scale λ_S")
    p.add_argument("--dt",         type=float, default=0.28, help="Pseudo-time step Δs")
    p.add_argument("--n-agents",   type=int,   default=14,   help="Initial swarm size")
    p.add_argument("--outdir",     type=str,   default="figures", help="Output directory")
    p.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    p.add_argument("--verbose",    action="store_true")
    p.add_argument("--list-targets", action="store_true",
                   help="Print available targets and exit")
    return p.parse_args(argv)


def run_single(target_name: str, args) -> dict:
    """Run SRFL on a single named target. Returns results dict."""
    info   = TARGETS[target_name]
    x      = np.linspace(-np.pi, np.pi, args.n)
    target = info["fn"](x)
    lam_sched = np.logspace(
        np.log10(args.lam0), np.log10(args.lam1), args.steps)

    print(f"\n  Target : {info['label']}")
    print(f"  Grid   : N={args.n},  S={args.steps}")
    print(f"  Scale  : λ ∈ [{lam_sched[-1]:.4f}, {lam_sched[0]:.2f}]")

    # Field evolution
    engine = SRFLField(x, target, lam_sched, dt=args.dt)
    fields, errors = engine.run(verbose=args.verbose)
    rate = engine.convergence_rate(errors)
    print(f"  L²err  : {errors[0]:.4f} → {errors[-1]:.4f}  (rate r≈{rate:.2f})")

    # Swarm
    swarm = Swarm(x, n_init=args.n_agents)
    for k, (phi, lam) in enumerate(zip(fields[1:], lam_sched[1:]), 1):
        swarm.step(phi, target, lam, k)
    evsum = swarm.event_summary()
    print(f"  Events : spawn={evsum['spawn']}  merge={evsum['merge']}"
          f"  annihilate={evsum['annihilate']}")

    # Action
    action = ActionFunctional(x, lam_sched, target)
    A = action.total(fields)
    print(f"  𝒜      : data={A['data']:.4f}  scale={A['scale']:.4f}"
          f"  total={A['total']:.4f}")

    # Defects
    alg     = DefectAlgebra(x)
    dx      = float(x[1] - x[0])
    defects = alg.detect_from_curvature(x, fields[-1], dx)
    print(f"  Defects: {len(defects)} detected — "
          + ", ".join(type(d).__name__ for d in defects))

    # Figures
    if not args.no_figures:
        os.makedirs(args.outdir, exist_ok=True)
        _save_figure(target_name, x, target, fields, errors,
                     lam_sched, swarm, args.outdir)
        print(f"  Figure : {args.outdir}/{target_name}_srfl.png")

    return {
        "target"     : target_name,
        "fields"     : fields,
        "errors"     : errors,
        "swarm"      : swarm,
        "action"     : A,
        "defects"    : defects,
        "conv_rate"  : rate,
    }


def _save_figure(name, x, target, fields, errors, lam_sched, swarm, outdir):
    import matplotlib.pyplot as plt

    sel  = np.linspace(0, len(fields) - 1, 8, dtype=int)
    cmap = plt.cm.viridis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Field evolution
    ax = axes[0]
    for k, idx in enumerate(sel):
        c = k / (len(sel) - 1)
        ax.plot(x, fields[idx], color=cmap(0.9 - 0.75*c),
                lw=1.7, alpha=0.9, label=f"λ={lam_sched[idx]:.3f}")
    ax.plot(x, target, "r--", lw=1.5, alpha=0.6, label="f(x)")
    ax.set_title("Field Evolution Φ(x,λ)")
    ax.set_xlabel("x"); ax.set_ylabel("Φ")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.22)

    # Convergence
    ax = axes[1]
    ax.semilogy(lam_sched, errors, "b-o", ms=3, lw=2)
    ax.invert_xaxis()
    ax.set_xlabel("λ  (→ fine)"); ax.set_ylabel("L² error")
    ax.set_title("Convergence ‖Φ − f‖_{L²}")
    ax.grid(True, which="both", alpha=0.25)

    # Swarm
    ax  = axes[2]
    n_h = len(swarm.history)
    for s, pos in enumerate(swarm.history):
        c = s / max(n_h - 1, 1)
        ax.scatter(pos, np.full(len(pos), s), s=12,
                   color=cmap(1 - c), alpha=0.65)
    for (s, etype, ex) in swarm.events:
        m = {"spawn":"^","merge":"D","annihilate":"x"}[etype]
        c = {"spawn":"lime","merge":"gold","annihilate":"red"}[etype]
        ax.scatter(ex, s, marker=m, s=80, color=c, zorder=5)
    ax.set_xlabel("x"); ax.set_ylabel("Scale step s")
    ax.set_title("Swarm Dynamics")
    ax.grid(True, alpha=0.2)

    plt.suptitle(f"SRFL — {TARGETS[name]['label']}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{outdir}/{name}_srfl.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)

    if args.list_targets:
        print("\nAvailable SRFL targets:")
        for k, v in TARGETS.items():
            print(f"  {k:<12} {v['label']}")
        print()
        return

    print("=" * 60)
    print("  Swarm Renormalization Field Learning (SRFL)")
    print("  Author: Bishal Neupane <cosmobishal@gmail.com>")
    print("=" * 60)

    if args.target in ("both", "all"):
        names = list(TARGETS.keys())
    else:
        names = [args.target]

    results = {}
    for name in names:
        results[name] = run_single(name, args)

    print("\n" + "=" * 60)
    print("  SRFL run complete.")
    print("=" * 60)
    return results


if __name__ == "__main__":
    main()
