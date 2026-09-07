from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .solver import SRFLConfig, SRFLSolver


def target_function(name: str, x: np.ndarray) -> np.ndarray:
    if name == "step":
        return (x >= 0).astype(float)
    if name == "oscillatory":
        out = np.zeros_like(x)
        mask = x != 0
        out[mask] = x[mask] * np.sin(1.0 / x[mask])
        return out
    if name == "sine":
        return np.sin(2.0 * x)
    if name == "gaussian":
        return np.exp(-2.0 * x * x)
    raise ValueError(f"unknown target '{name}'")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="srfl", description="Run Swarm Renormalization Field Learning")
    parser.add_argument("--target", choices=["step", "oscillatory", "sine", "gaussian"], default="step")
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--scales", type=int, default=64)
    parser.add_argument("--lam-max", type=float, default=1.0)
    parser.add_argument("--lam-min", type=float, default=0.02)
    parser.add_argument("--dt", type=float, default=0.4)
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    if args.n < 32:
        parser.error("--n must be at least 32")
    x = np.linspace(-np.pi, np.pi, args.n)
    y = target_function(args.target, x)
    cfg = SRFLConfig(scales=args.scales, lam_max=args.lam_max, lam_min=args.lam_min,
                     dt=args.dt, n_agents=args.agents)
    result = SRFLSolver(cfg).fit(x, y)

    args.out.mkdir(parents=True, exist_ok=True)
    result.save_npz(args.out / f"{args.target}.npz")
    summary = {
        "target": args.target,
        "n": args.n,
        "scales": args.scales,
        "initial_rmse": result.initial_error,
        "final_rmse": result.final_error,
        "relative_improvement": result.improvement,
        "peak_agents": int(np.max(result.agent_counts)),
        "events": result.events,
    }
    (args.out / f"{args.target}.json").write_text(json.dumps(summary, indent=2))

    if not args.no_figure:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise SystemExit("matplotlib is required for figures. Install srfl[plot]") from exc
        fig, ax = plt.subplots(figsize=(9, 5))
        picks = np.linspace(0, len(result.fields) - 1, min(8, len(result.fields))).astype(int)
        for idx in picks:
            ax.plot(x, result.fields[idx], alpha=0.55, linewidth=1.0,
                    label=f"lambda={result.scales[idx]:.3g}")
        ax.plot(x, y, linewidth=2.2, label="target")
        ax.set_xlabel("x")
        ax.set_ylabel("field")
        ax.set_title(f"SRFL field flow for {args.target}")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out / f"{args.target}.png", dpi=170)
        plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != "events"}, indent=2))
    return 0
