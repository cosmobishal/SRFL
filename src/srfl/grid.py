from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Grid1D:
    """Validated uniform one dimensional domain."""

    x: np.ndarray

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float)
        if x.ndim != 1 or x.size < 8:
            raise ValueError("x must be a one dimensional array with at least 8 points")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must contain only finite values")
        dx = np.diff(x)
        if np.any(dx <= 0):
            raise ValueError("x must be strictly increasing")
        if not np.allclose(dx, dx[0], rtol=1e-10, atol=1e-14):
            raise ValueError("x must be uniformly spaced")
        object.__setattr__(self, "x", x.copy())

    @property
    def n(self) -> int:
        return int(self.x.size)

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])

    @property
    def length(self) -> float:
        return float(self.x[-1] - self.x[0])

    def check_values(self, values: np.ndarray, name: str = "values") -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.shape != self.x.shape:
            raise ValueError(f"{name} must have shape {self.x.shape}, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")
        return values
