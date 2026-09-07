# Reproducibility results

These values are local smoke benchmarks for the release.
They are not claims of state of the art performance.
The purpose is to make the examples measurable and repeatable.

| Target | Grid | Scales | Initial RMSE | Final RMSE | Improvement |
|---|---:|---:|---:|---:|---:|
| step | 512 | 48 | 0.182778 | 0.010571 | 94.22% |
| oscillatory | 512 | 48 | 0.211898 | 0.008788 | 95.85% |
| sine | 512 | 48 | 0.498753 | 0.012292 | 97.54% |
| gaussian | 512 | 48 | 0.187644 | 0.003260 | 98.26% |

The benchmark script is `benchmarks/benchmark_targets.py`.
Run it from the repository root with

```bash
PYTHONPATH=src python benchmarks/benchmark_targets.py
```

The exact timing depends on the machine and is intentionally not treated as a scientific result.
