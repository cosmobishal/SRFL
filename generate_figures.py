"""
scripts/generate_figures.py
============================
Standalone script to reproduce all 7 publication figures from the
SRFL paper without running the full experiment pipeline.

Usage
-----
    python scripts/generate_figures.py
    python scripts/generate_figures.py --outdir figures --n 512 --steps 70
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srfl import SRFLField, Swarm, ScaleProjection


def parse_args():
    p = argparse.ArgumentParser(description="SRFL — Publication Figure Generator")
    p.add_argument("--n",      type=int, default=512,   help="Grid size")
    p.add_argument("--steps",  type=int, default=70,    help="Scale steps")
    p.add_argument("--outdir", type=str, default="figures", help="Output dir")
    p.add_argument("--dpi",    type=int, default=150,   help="Figure DPI")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("Generating SRFL publication figures …")
    print(f"  Grid: N={args.n},  Steps: {args.steps},  DPI: {args.dpi}")
    print(f"  Output: {args.outdir}/\n")

    # ── Shared setup ────────────────────────────────────────────────
    np.random.seed(42)
    x          = np.linspace(-np.pi, np.pi, args.n)
    dx         = float(x[1] - x[0])
    lam_sched  = np.logspace(0, -1.85, args.steps)

    def step_fn(x):
        return np.where(x >= 0, 1.0, 0.0).astype(float)

    def osc_fn(x):
        out = np.zeros_like(x, dtype=float)
        nz  = x != 0
        out[nz] = x[nz] * np.sin(1.0 / x[nz])
        return out

    t_step = step_fn(x)
    t_osc  = osc_fn(x)

    # Run field evolutions
    print("  Running step-function field evolution …")
    eng_step = SRFLField(x, t_step, lam_sched)
    f_step, e_step = eng_step.run()

    print("  Running oscillatory field evolution …")
    eng_osc  = SRFLField(x, t_osc,  lam_sched)
    f_osc,  e_osc  = eng_osc.run()

    # Run swarms
    print("  Running swarms …")
    sw_step = Swarm(x, n_init=14)
    for k, (phi, lam) in enumerate(zip(f_step[1:], lam_sched[1:]), 1):
        sw_step.step(phi, t_step, lam, k)

    sw_osc = Swarm(x, n_init=14)
    for k, (phi, lam) in enumerate(zip(f_osc[1:], lam_sched[1:]), 1):
        sw_osc.step(phi, t_osc, lam, k)

    print()

    # ── Figure 1: field_evolution ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sel  = np.linspace(0, args.steps - 1, 8, dtype=int)
    cmap = plt.cm.viridis
    for ax, fields, tgt, ttl in zip(
        axes,
        [f_step, f_osc],
        [t_step, t_osc],
        ["Step function $f(x)=H(x)$", r"Oscillatory $f(x)=x\sin(1/x)$"]
    ):
        for k, idx in enumerate(sel):
            c = k / (len(sel) - 1)
            ax.plot(x, fields[idx], color=cmap(0.9 - 0.75*c),
                    lw=1.7, label=f"$\\lambda={lam_sched[idx]:.3f}$", alpha=0.92)
        ax.plot(x, tgt, "r--", lw=1.5, alpha=0.65, label="Target $f(x)$")
        ax.set_title(ttl, fontsize=12); ax.set_xlabel("$x$")
        ax.set_ylabel(r"$\Phi(x,\lambda)$")
        ax.legend(fontsize=7, ncol=2, loc="lower right")
        ax.grid(True, alpha=0.22); ax.set_xlim(x[0], x[-1])
    plt.suptitle(
        r"SRFL Field Evolution: $\partial_s\Phi = \int K\,[f-\Phi]\,dx' + \mathcal{S}[\Phi]$",
        fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"{args.outdir}/field_evolution.png", dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print("  ✓  field_evolution.png")

    # ── Figure 2: swarm_structure ────────────────────────────────────
    n_plot = min(len(sw_step.history), args.steps)
    fig, ax = plt.subplots(figsize=(12, 7))
    for s in range(0, n_plot, 2):
        pos = sw_step.history[s]
        c   = s / n_plot
        ax.scatter(pos, np.full(len(pos), s), s=22,
                   color=plt.cm.plasma(1 - c), alpha=0.75, zorder=3)
    for s in range(0, n_plot - 1, 5):
        pos = sw_step.history[s]
        for i, xi in enumerate(pos):
            for j, xj in enumerate(pos):
                if i >= j: continue
                d = abs(xi - xj)
                if d < 0.8:
                    ax.plot([xi, xj], [s, s], "-", color="steelblue",
                            alpha=max(0.04, 0.35 - d * 0.4), lw=0.9, zorder=2)
    for (s, etype, ex) in sw_step.events:
        if s >= n_plot: continue
        m = {"spawn":"^","merge":"D","annihilate":"x"}[etype]
        c = {"spawn":"lime","merge":"gold","annihilate":"red"}[etype]
        ax.scatter(ex, s, marker=m, s=90, color=c, zorder=5)
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("Scale step $s$  (coarse $\\to$ fine)", fontsize=12)
    ax.set_title("SRFL Swarm Structure in $(x,s)$ Space — Step Function", fontsize=13)
    ax.grid(True, alpha=0.18)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ticks = np.linspace(0, n_plot - 1, 6, dtype=int)
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([f"$\\lambda={lam_sched[t]:.3f}$" for t in ticks], fontsize=8)
    ax2.set_ylabel("Scale $\\lambda$")
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/swarm_structure.png", dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print("  ✓  swarm_structure.png")

    # ── Figure 3: nonlocal_kernel ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    lv_vals = [1.0, 0.5, 0.2, 0.08, 0.03]
    ck = plt.cm.coolwarm(np.linspace(0.05, 0.95, len(lv_vals)))
    ax = axes[0]
    for lv, c in zip(lv_vals, ck):
        K = np.exp(-x**2 / (2 * lv**2)); K /= K.max()
        ax.plot(x, K, color=c, lw=2.1, label=f"$\\lambda={lv}$")
    ax.axvline(0, color="gray", ls="--", alpha=0.5)
    ax.set_title("$K(x,\\,x'=0,\\,\\lambda)$", fontsize=12)
    ax.set_xlabel("$x$"); ax.set_ylabel("$K$ (normalised)"); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax = axes[1]
    step_s = max(1, args.n // 120); xs = x[::step_s]
    K2d = np.exp(-(xs[:,None] - xs[None,:])**2 / (2 * 0.25**2))
    im = ax.imshow(K2d, extent=[xs[0],xs[-1],xs[-1],xs[0]],
                   cmap="hot", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("$K(x,x',\\lambda=0.25)$", fontsize=12)
    ax.set_xlabel("$x'$"); ax.set_ylabel("$x$")
    ax = axes[2]
    lv_r = np.logspace(-2, 0.1, 60)
    ax.loglog(lv_r, 2 * np.sqrt(2 * np.log(2)) * lv_r, "b-", lw=2.5,
              label="FWHM $= 2\\sqrt{2\\ln 2}\\,\\lambda$")
    ax.loglog(lv_r, lv_r, "r--", lw=1.5, alpha=0.65, label="$y=\\lambda$")
    ax.set_xlabel("$\\lambda$"); ax.set_ylabel("Width")
    ax.set_title("Receptive Field Width vs $\\lambda$")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.25)
    plt.suptitle("Non-local Kernel $K(x,x',\\lambda)$", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(f"{args.outdir}/nonlocal_kernel.png", dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print("  ✓  nonlocal_kernel.png")

    # ── Figure 4: step_defect ────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    stage_ids = np.linspace(0, args.steps - 1, 6, dtype=int)
    for ax, sid in zip(axes.flat, stage_ids):
        phi = f_step[sid]; lv = lam_sched[sid]
        d2  = np.gradient(np.gradient(phi, dx), dx)
        thr = np.percentile(np.abs(d2), 90)
        ax.plot(x, phi, "b-", lw=2.0, label=r"$\Phi$", zorder=3)
        ax.plot(x, t_step, "k--", lw=1.4, alpha=0.55, label="$H(x)$")
        ax.fill_between(x, -0.4, 1.4, where=np.abs(d2) > thr,
                        color="crimson", alpha=0.16, label="Defect zone")
        ax.set_title(f"$\\lambda={lv:.4f}$")
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.35, 1.35)
        ax.set_xlabel("$x$"); ax.set_ylabel(r"$\Phi$")
        ax.legend(fontsize=7.5, loc="upper left"); ax.grid(True, alpha=0.2)
    plt.suptitle(r"Step Defect $\hat{S}_{x_0}$: Zone Contracts to Discontinuity $x=0$",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{args.outdir}/step_defect.png", dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print("  ✓  step_defect.png")

    # ── Figure 5: oscillation_defect ─────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, sid in zip(axes.flat, stage_ids):
        phi = f_osc[sid]; lv = lam_sched[sid]
        d2  = np.gradient(np.gradient(phi, dx), dx)
        thr = np.percentile(np.abs(d2), 88)
        ax.plot(x, phi, color="purple", lw=2.0, label=r"$\Phi$", zorder=3)
        ax.plot(x, t_osc, "k--", lw=1.4, alpha=0.55, label=r"$x\sin(1/x)$")
        ax.fill_between(x, -1.6, 1.6, where=np.abs(d2) > thr,
                        color="darkorange", alpha=0.20, label=r"Osc. defect $\hat{O}$")
        ax.set_title(f"$\\lambda={lv:.4f}$")
        ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.6, 1.6)
        ax.set_xlabel("$x$"); ax.set_ylabel(r"$\Phi$")
        ax.legend(fontsize=7.5, loc="upper right"); ax.grid(True, alpha=0.2)
    plt.suptitle(r"Oscillatory Defect $\hat{O}_\varepsilon$: Tracks Singularity at $x=0$",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{args.outdir}/oscillation_defect.png", dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print("  ✓  oscillation_defect.png")

    # ── Figure 6: swarm_dynamics ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, sw, tname in zip(
        axes,
        [sw_step, sw_osc],
        ["Step $H(x)$", r"Oscillatory $x\sin(1/x)$"]
    ):
        n_s = min(len(sw.history), args.steps)
        counts = [len(h) for h in sw.history[:n_s]]
        for s in range(n_s):
            c = plt.cm.RdYlGn(0.15 + 0.7 * s / n_s)
            ax.scatter(sw.history[s], np.full(len(sw.history[s]), s),
                       s=14, color=c, alpha=0.65, zorder=3)
        for (s, etype, ex) in sw.events:
            if s >= n_s: continue
            m = {"spawn":"*","merge":"D","annihilate":"X"}[etype]
            c = {"spawn":"cyan","merge":"yellow","annihilate":"red"}[etype]
            ec= {"spawn":"blue","merge":"orange","annihilate":"darkred"}[etype]
            ax.scatter(ex, s, marker=m, s=140 if etype=="spawn" else 75,
                       color=c, edgecolors=ec, lw=0.8, zorder=5)
        ax_in = ax.inset_axes([0.65, 0.05, 0.32, 0.28])
        ax_in.plot(np.arange(n_s), counts[:n_s], "k-", lw=1.4)
        ax_in.set_title("Agent count", fontsize=7)
        ax_in.tick_params(labelsize=6); ax_in.grid(True, alpha=0.3)
        ax.set_xlabel("$x$"); ax.set_ylabel("Scale step $s$")
        ax.set_title(tname); ax.grid(True, alpha=0.18)
    plt.suptitle("SRFL Swarm Dynamics: Spawn / Merge / Annihilate", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{args.outdir}/swarm_dynamics.png", dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print("  ✓  swarm_dynamics.png")

    # ── Figure 7: multiscale_flow ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes[0]
    field_mat = np.array(f_step)
    im = ax.imshow(field_mat, extent=[x[0],x[-1],args.steps-1,0],
                   cmap="RdBu_r", aspect="auto", vmin=-1.2, vmax=1.2)
    plt.colorbar(im, ax=ax, label=r"$\Phi(x,\lambda)$", fraction=0.046, pad=0.04)
    ax.set_xlabel("$x$"); ax.set_ylabel("Scale step $s$")
    ax.set_title(r"Full Scale Map $\Phi(x,\lambda(s))$", fontsize=11)
    ax = axes[1]
    ids  = [3, 25, min(55, args.steps - 1)]
    cols = ["steelblue", "darkorange", "darkgreen"]
    labs = [f"$\\lambda_1={lam_sched[ids[0]]:.3f}$",
            f"$\\lambda_2={lam_sched[ids[1]]:.3f}$",
            f"$\\lambda_3={lam_sched[ids[2]]:.3f}$"]
    for i_id, c, lab in zip(ids, cols, labs):
        ax.plot(x, f_step[i_id], color=c, lw=2.4, label=lab)
    ax.plot(x, t_step, "k--", lw=1.5, alpha=0.6, label="$H(x)$")
    probe = 0.6
    for (ia, ib) in [(ids[0], ids[1]), (ids[1], ids[2])]:
        ya = float(np.interp(probe, x, f_step[ia]))
        yb = float(np.interp(probe, x, f_step[ib]))
        ax.annotate("", xy=(probe, yb), xytext=(probe, ya),
                    arrowprops=dict(arrowstyle="->", color="gray",
                                   lw=1.8, mutation_scale=14))
    ax.set_title(r"Projection $\Pi(\lambda_1 \to \lambda_2 \to \lambda_3)$")
    ax.set_xlabel("$x$"); ax.set_ylabel(r"$\Phi$")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    ax = axes[2]
    ax.semilogy(lam_sched, e_step, "b-o", ms=3.5, lw=1.8, label="Step $H(x)$")
    ax.semilogy(lam_sched, e_osc,  "r-s", ms=3.5, lw=1.8, label=r"Osc $x\sin(1/x)$")
    ax.invert_xaxis()
    ax.set_xlabel("$\\lambda$"); ax.set_ylabel(r"$\|\Phi-f\|_{L^2}$")
    ax.set_title("Error vs $\\lambda$")
    ax.legend(fontsize=9); ax.grid(True, which="both", alpha=0.25)
    plt.suptitle(r"Multi-scale Flow: $\Pi(\lambda_1\to\lambda_2)$, Scale Consistency",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(f"{args.outdir}/multiscale_flow.png", dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print("  ✓  multiscale_flow.png")

    print(f"\n  All 7 figures saved to '{args.outdir}/'")


if __name__ == "__main__":
    main()
