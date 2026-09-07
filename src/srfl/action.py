from __future__ import annotations

from typing import Optional

import numpy as np


class ActionFunctional:
    """Finite scale diagnostics for an SRFL trajectory."""

    def __init__(self, x: np.ndarray, scales: np.ndarray, target: np.ndarray,
                 beta: float = 1.0, symmetry=None) -> None:
        self.x = np.asarray(x, dtype=float)
        self.scales = np.asarray(scales, dtype=float)
        self.target = np.asarray(target, dtype=float)
        if self.x.ndim != 1 or self.x.size < 2:
            raise ValueError("x must be one dimensional")
        if self.scales.ndim != 1 or self.scales.size < 1 or np.any(self.scales <= 0):
            raise ValueError("scales must be positive")
        if self.target.shape != self.x.shape:
            raise ValueError("target and x must have the same shape")
        self.beta = float(beta)
        self.symmetry = symmetry

    def data(self, fields: list[np.ndarray]) -> float:
        vals = [np.mean((phi - self.target) ** 2) * np.exp(-lam) / max(lam, 1e-12)
                for phi, lam in zip(fields, self.scales)]
        return float(np.mean(vals))

    def scale_consistency(self, fields: list[np.ndarray], kernel_factory) -> float:
        """Compare each fine field with the Gaussian projection of its coarse neighbor."""
        terms: list[float] = []
        for i in range(len(fields) - 1):
            coarse_lam = float(self.scales[i])
            fine_lam = float(self.scales[i + 1])
            if coarse_lam <= fine_lam:
                continue
            blur = np.sqrt(max(coarse_lam * coarse_lam - fine_lam * fine_lam, 0.0))
            projected_fine = kernel_factory(blur).convolve(fields[i + 1]) if blur > 0 else fields[i + 1]
            terms.append(float(np.mean((fields[i] - projected_fine) ** 2)))
        return float(np.mean(terms)) if terms else 0.0

    def symmetry_term(self, fields: list[np.ndarray]) -> float:
        if self.symmetry is None:
            return 0.0
        return float(np.mean([np.mean((phi - self.symmetry(phi)) ** 2) for phi in fields]))

    def complexity(self, defect_norms: Optional[list[float]] = None) -> float:
        if defect_norms is None:
            return 0.0
        return float(np.mean([(lam ** self.beta) * norm ** 2
                              for lam, norm in zip(self.scales, defect_norms)]))

    def total(self, fields: list[np.ndarray], kernel_factory, defect_norms=None) -> dict[str, float]:
        data = self.data(fields)
        scale = self.scale_consistency(fields, kernel_factory)
        symmetry = self.symmetry_term(fields)
        complexity = self.complexity(defect_norms)
        return {"data": data, "scale": scale, "symmetry": symmetry,
                "complexity": complexity, "total": data + scale + symmetry + complexity}
