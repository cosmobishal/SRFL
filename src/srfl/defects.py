from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np


@dataclass(frozen=True)
class DefectCandidate:
    kind: str
    center: float
    width: float
    score: float
    sign: float
    amplitude: float = 0.0

    def with_amplitude(self, amplitude: float) -> "DefectCandidate":
        return DefectCandidate(self.kind, self.center, self.width, self.score, self.sign, float(amplitude))


class DefectAtom:
    name = "base"

    def evaluate(self, x: np.ndarray, candidate: DefectCandidate) -> np.ndarray:
        raise NotImplementedError


class StepAtom(DefectAtom):
    name = "step"

    def evaluate(self, x: np.ndarray, candidate: DefectCandidate) -> np.ndarray:
        return (x >= candidate.center).astype(float)


class OscillatoryAtom(DefectAtom):
    name = "oscillatory"

    def evaluate(self, x: np.ndarray, candidate: DefectCandidate) -> np.ndarray:
        width = max(candidate.width, np.finfo(float).eps)
        z = x - candidate.center
        out = np.zeros_like(x, dtype=float)
        mask = np.abs(z) <= width
        oscillatory = mask & (z != 0.0)
        out[oscillatory] = z[oscillatory] * np.sin(1.0 / z[oscillatory])
        return out


class PiecewiseAtom(DefectAtom):
    name = "piecewise"

    def evaluate(self, x: np.ndarray, candidate: DefectCandidate) -> np.ndarray:
        half = max(candidate.width, np.finfo(float).eps) * 0.5
        return ((x >= candidate.center - half) & (x < candidate.center + half)).astype(float)


class GaussianBumpAtom(DefectAtom):
    name = "bump"

    def evaluate(self, x: np.ndarray, candidate: DefectCandidate) -> np.ndarray:
        width = max(candidate.width, np.finfo(float).eps)
        return np.exp(-0.5 * ((x - candidate.center) / width) ** 2)


class DefectRegistry:
    """Registry of defect atoms used by the adaptive projection stage."""

    def __init__(self, atoms: Optional[Iterable[DefectAtom]] = None) -> None:
        default = [StepAtom(), OscillatoryAtom(), PiecewiseAtom(), GaussianBumpAtom()]
        self._atoms = {atom.name: atom for atom in (default if atoms is None else atoms)}
        if not self._atoms:
            raise ValueError("at least one defect atom is required")

    def register(self, atom: DefectAtom) -> None:
        if not getattr(atom, "name", None):
            raise ValueError("defect atom must define a non empty name")
        self._atoms[atom.name] = atom

    def evaluate(self, x: np.ndarray, candidate: DefectCandidate) -> np.ndarray:
        try:
            atom = self._atoms[candidate.kind]
        except KeyError as exc:
            raise KeyError(f"unknown defect kind '{candidate.kind}'") from exc
        return atom.evaluate(x, candidate)

    def names(self) -> tuple[str, ...]:
        return tuple(self._atoms)


def _robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return float(1.4826 * mad + 1e-12)


def _local_zero_crossing_density(values: np.ndarray, radius: int) -> np.ndarray:
    signs = np.sign(values)
    crossings = (signs[:-1] * signs[1:] < 0).astype(float)
    result = np.zeros_like(values, dtype=float)
    for i in range(values.size):
        lo = max(0, i - radius)
        hi = min(crossings.size, i + radius)
        span = max(hi - lo, 1)
        result[i] = crossings[lo:hi].sum() / span
    return result


class DefectDetector:
    """Detect candidate singular regions from scale normalized residual geometry."""

    def __init__(self, sensitivity: float = 4.0, max_candidates: int = 6,
                 min_separation: float = 0.0) -> None:
        if sensitivity <= 0 or max_candidates < 1 or min_separation < 0:
            raise ValueError("invalid defect detector parameters")
        self.sensitivity = float(sensitivity)
        self.max_candidates = int(max_candidates)
        self.min_separation = float(min_separation)

    def detect(self, x: np.ndarray, field: np.ndarray, residual: np.ndarray,
               lam: float) -> list[DefectCandidate]:
        if lam <= 0:
            raise ValueError("lam must be positive")
        x = np.asarray(x, dtype=float)
        field = np.asarray(field, dtype=float)
        residual = np.asarray(residual, dtype=float)
        dx = float(x[1] - x[0])
        d1 = np.gradient(field, dx)
        d2 = np.gradient(d1, dx)
        curvature = np.abs((lam ** 2) * d2)
        curvature_score = curvature / _robust_scale(curvature)
        residual_score = np.abs(residual) / _robust_scale(residual)
        feature_score = np.maximum(curvature_score, residual_score)
        crossings = _local_zero_crossing_density(residual, max(2, int(round(lam / dx))))
        peaks = np.where(feature_score >= self.sensitivity)[0]
        if peaks.size == 0:
            peaks = np.argsort(feature_score)[-min(self.max_candidates, feature_score.size):]
        order = peaks[np.argsort(feature_score[peaks])[::-1]]

        candidates: list[DefectCandidate] = []
        for idx in order:
            center = float(x[idx])
            if any(abs(center - c.center) < max(self.min_separation, 1.5 * dx) for c in candidates):
                continue
            local = max(2, int(round(lam / dx)))
            lo = max(0, idx - local)
            hi = min(x.size, idx + local + 1)
            local_cross = float(np.mean(crossings[lo:hi]))
            local_amp = float(np.max(np.abs(residual[lo:hi]))) if hi > lo else 0.0
            sign = float(np.sign(np.sum(residual[lo:hi])) or 1.0)
            if local_cross > 0.08 and local_amp > _robust_scale(residual):
                kind = "oscillatory"
                width = min(max(lam, 2.0 * dx), 4.0 * lam)
            elif feature_score[idx] >= 1.5 * self.sensitivity:
                kind = "step"
                width = max(2.0 * dx, 0.5 * lam)
            else:
                kind = "piecewise"
                width = max(2.0 * dx, lam)
            candidates.append(DefectCandidate(kind, center, width,
                                               float(feature_score[idx]), sign))
            if len(candidates) >= self.max_candidates:
                break
        return candidates


class DefectAlgebra:
    """Small algebra used for diagnostics and custom extensions."""

    @staticmethod
    def l1_norm(candidates: Iterable[DefectCandidate]) -> float:
        return float(sum(abs(c.amplitude) for c in candidates))

    @staticmethod
    def compose(a: DefectCandidate, b: DefectCandidate, tol: float = 1e-12) -> DefectCandidate:
        if a.kind != b.kind or abs(a.center - b.center) > tol:
            raise ValueError("only coincident defects of the same type can be composed")
        return DefectCandidate(a.kind, a.center, max(a.width, b.width),
                               max(a.score, b.score), a.sign, a.amplitude + b.amplitude)
