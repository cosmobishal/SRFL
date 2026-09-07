from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Optional

from .grid import Grid1D


@dataclass(frozen=True)
class GaussianKernel:
    """Scale indexed Gaussian operator for a uniform one dimensional grid.

    The default boundary is reflective. This avoids the artificial wrap around
    introduced by a periodic FFT when the input is defined on a finite interval.
    The ``wrap`` mode is available when periodic geometry is intentional.
    """

    grid: Grid1D
    lam: float
    boundary: str = "reflect"

    def __post_init__(self) -> None:
        if not np.isfinite(self.lam) or self.lam <= 0:
            raise ValueError("lam must be finite and positive")
        if self.boundary not in {"reflect", "wrap"}:
            raise ValueError("boundary must be 'reflect' or 'wrap'")

    @property
    def sigma_pixels(self) -> float:
        return float(self.lam / self.grid.dx)

    @property
    def fwhm(self) -> float:
        return float(2.0 * np.sqrt(2.0 * np.log(2.0)) * self.lam)

    def convolve(self, values: np.ndarray) -> np.ndarray:
        values = self.grid.check_values(values)
        if self.sigma_pixels < 1e-8:
            return values.copy()
        mode = "reflect" if self.boundary == "reflect" else "wrap"
        return gaussian_filter1d(values, sigma=self.sigma_pixels, mode=mode, truncate=5.0)

    def matrix(self, subsample: int = 1) -> np.ndarray:
        if subsample < 1 or int(subsample) != subsample:
            raise ValueError("subsample must be a positive integer")
        xs = self.grid.x[::subsample]
        delta = xs[:, None] - xs[None, :]
        mat = np.exp(-0.5 * (delta / self.lam) ** 2)
        row_sum = mat.sum(axis=1, keepdims=True)
        return mat / np.maximum(row_sum, np.finfo(float).eps)

    def interaction(self, xi: float, xj: float, lam_i: Optional[float] = None,
                    lam_j: Optional[float] = None, sigma_lam: float = 0.1) -> float:
        if sigma_lam <= 0:
            raise ValueError("sigma_lam must be positive")
        spatial = np.exp(-0.5 * ((xi - xj) / self.lam) ** 2)
        if lam_i is None or lam_j is None:
            scale = 1.0
        else:
            scale = np.exp(-0.5 * ((lam_i - lam_j) / sigma_lam) ** 2)
        return float(spatial * scale)

    def update(self, lam: float) -> "GaussianKernel":
        return GaussianKernel(self.grid, lam, self.boundary)

    def __repr__(self) -> str:
        return f"GaussianKernel(n={self.grid.n}, lam={self.lam:.6g}, boundary='{self.boundary}')"
