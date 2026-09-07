from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from srfl import SRFLConfig, SRFLSolver


def make_targets(x: np.ndarray) -> dict[str, np.ndarray]:
    osc = np.zeros_like(x)
    mask = x != 0
    osc[mask] = x[mask] * np.sin(1.0 / x[mask])
    return {
        "step": (x >= 0).astype(float),
        "oscillatory": osc,
        "sine": np.sin(2.0 * x),
        "gaussian": np.exp(-2.0 * x * x),
    }


def main() -> None:
    x = np.linspace(-np.pi, np.pi, 512)
    config = SRFLConfig(scales=48, lam_max=0.9, lam_min=0.025, dt=0.4, n_agents=8)
    rows = []
    for name, target in make_targets(x).items():
        start = perf_counter()
        result = SRFLSolver(config).fit(x, target)
        elapsed = perf_counter() - start
        rows.append({
            "target": name,
            "grid": x.size,
            "scales": config.scales,
            "initial_rmse": result.initial_error,
            "final_rmse": result.final_error,
            "improvement_percent": 100.0 * result.improvement,
            "max_agents": int(result.agent_counts.max()),
            "events": len(result.events),
            "seconds": elapsed,
        })
    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    (out / "benchmark_targets.json").write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(f"{row['target']:12s}  rmse {row['initial_rmse']:.6f} -> {row['final_rmse']:.6f}  "
              f"improvement {row['improvement_percent']:.2f}%  "
              f"agents {row['max_agents']:2d}  time {row['seconds']:.3f}s")


if __name__ == "__main__":
    main()
