"""
srfl.defects
============
Defect algebra  𝒟 = span{Ŝ, Ô, Ĉ}.

Three canonical defect operators and their algebra under composition.

Ŝ_{x₀,α}  — step defect:           φ(x) + α·H(x - x₀)
Ô_{ε,β}   — oscillatory defect:    φ(x) + β·χ_{|x|<ε}·x·sin(1/x)
Ĉ_{I,a}   — conditional defect:    φ(x) + Σ aₖ·χ_{Iₖ}(x)

The algebra is associative under composition but non-commutative when
singularity points coincide.
"""

import numpy as np
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
class StepDefect:
    """
    Step defect operator  Ŝ_{x₀, α}.

    Action:  φ(x) ↦ φ(x) + α · H(x − x₀)

    Parameters
    ----------
    x0    : float   — discontinuity location
    alpha : float   — amplitude
    """

    def __init__(self, x0: float = 0.0, alpha: float = 1.0):
        self.x0    = x0
        self.alpha = alpha

    def apply(self, x: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Apply the defect: φ ↦ φ + α·H(x − x₀)."""
        return phi + self.alpha * (x >= self.x0).astype(float)

    def field(self, x: np.ndarray) -> np.ndarray:
        """Return the defect field D(x) = α·H(x − x₀)."""
        return self.alpha * (x >= self.x0).astype(float)

    def norm(self) -> float:
        """𝒟-norm = |α|."""
        return abs(self.alpha)

    def compose(self, other: "StepDefect") -> "StepDefect":
        """Compose two step defects at the same location."""
        if not isinstance(other, StepDefect):
            raise TypeError("Can only compose StepDefect with StepDefect")
        if abs(self.x0 - other.x0) > 1e-12:
            raise ValueError("Composition defined only for same x0 in this implementation")
        return StepDefect(x0=self.x0, alpha=self.alpha + other.alpha)

    def __repr__(self) -> str:
        return f"StepDefect(x0={self.x0:.4f}, α={self.alpha:.4f})"


# ─────────────────────────────────────────────────────────────────────────────
class OscillatoryDefect:
    """
    Oscillatory defect operator  Ô_{ε, β}.

    Action:  φ(x) ↦ φ(x) + β · χ_{|x|<ε}(x) · x·sin(1/x)

    Captures the essential singularity structure of x·sin(1/x) at x=0.

    Parameters
    ----------
    eps  : float  — support half-width  ε > 0
    beta : float  — amplitude
    """

    def __init__(self, eps: float = 0.5, beta: float = 1.0):
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps  = eps
        self.beta = beta

    def apply(self, x: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Apply the defect: φ ↦ φ + β·χ_{|x|<ε}·x·sin(1/x)."""
        return phi + self.beta * self._osc_field(x)

    def field(self, x: np.ndarray) -> np.ndarray:
        """Return the defect field D(x) = β·χ_{|x|<ε}·x·sin(1/x)."""
        return self.beta * self._osc_field(x)

    def _osc_field(self, x: np.ndarray) -> np.ndarray:
        out  = np.zeros_like(x, dtype=float)
        mask = np.abs(x) < self.eps
        nz   = mask & (x != 0)
        out[nz] = x[nz] * np.sin(1.0 / x[nz])
        return out

    def norm(self) -> float:
        """𝒟-norm = |β|·ε  (L¹ mass proxy)."""
        return abs(self.beta) * self.eps

    def __repr__(self) -> str:
        return f"OscillatoryDefect(ε={self.eps:.4f}, β={self.beta:.4f})"


# ─────────────────────────────────────────────────────────────────────────────
class ConditionalDefect:
    """
    Conditional (piecewise-constant) defect  Ĉ_{I,a}.

    Action:  φ(x) ↦ φ(x) + Σₖ aₖ·χ_{Iₖ}(x)

    Parameters
    ----------
    intervals : list of (lo, hi) tuples  — partition elements Iₖ
    amplitudes: list of floats           — coefficients aₖ
    """

    def __init__(self, intervals: List[Tuple[float, float]],
                 amplitudes: List[float]):
        if len(intervals) != len(amplitudes):
            raise ValueError("intervals and amplitudes must have the same length")
        self.intervals  = intervals
        self.amplitudes = amplitudes

    def apply(self, x: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return phi + self.field(x)

    def field(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=float)
        for (lo, hi), a in zip(self.intervals, self.amplitudes):
            out[(x >= lo) & (x < hi)] += a
        return out

    def norm(self) -> float:
        """𝒟-norm = Σ|aₖ|."""
        return sum(abs(a) for a in self.amplitudes)

    def __repr__(self) -> str:
        pairs = [(f"[{lo:.2f},{hi:.2f}]", f"{a:.3f}")
                 for (lo, hi), a in zip(self.intervals, self.amplitudes)]
        return f"ConditionalDefect({pairs})"


# ─────────────────────────────────────────────────────────────────────────────
class DefectAlgebra:
    """
    The defect algebra  𝒟 over a spatial grid x.

    Supports:
      • Sequential composition  D₁ ∘ D₂
      • Commutator  [D₁, D₂] = D₁∘D₂ − D₂∘D₁  (as field difference)
      • 𝒟-norm  ‖D‖_𝒟
      • Auto-detection of defect type from curvature profile

    Parameters
    ----------
    x : np.ndarray
        Spatial grid.
    """

    def __init__(self, x: np.ndarray):
        self.x = x

    def compose_fields(self, D1, D2, phi: np.ndarray) -> np.ndarray:
        """Apply D1 then D2: (D2 ∘ D1)[φ]."""
        phi1 = D1.apply(self.x, phi)
        return D2.apply(self.x, phi1)

    def commutator_field(self, D1, D2,
                         phi: np.ndarray) -> np.ndarray:
        """
        Compute the commutator field:
            [D1, D2][φ](x) = (D1∘D2)[φ](x) − (D2∘D1)[φ](x)
        """
        d12 = self.compose_fields(D2, D1, phi)   # D1 ∘ D2
        d21 = self.compose_fields(D1, D2, phi)   # D2 ∘ D1
        return d12 - d21

    def total_norm(self, defects: list) -> float:
        """‖Σ Dᵢ‖_𝒟 = Σ ‖Dᵢ‖_𝒟."""
        return sum(d.norm() for d in defects)

    @staticmethod
    def detect_from_curvature(x: np.ndarray, phi: np.ndarray,
                               dx: float,
                               kappa: float = 1.5,
                               eps_osc: float = 0.3) -> list:
        """
        Auto-detect which defect types are active based on curvature profile.

        Rules:
        - High curvature at isolated point → StepDefect
        - High curvature in oscillatory cluster near x=0 → OscillatoryDefect
        - High curvature on interval → ConditionalDefect

        Returns list of instantiated defect objects.
        """
        d2    = np.gradient(np.gradient(phi, dx), dx)
        above = np.abs(d2) > kappa
        detected = []

        # Find connected high-curvature regions
        transitions = np.diff(above.astype(int))
        starts = np.where(transitions == 1)[0] + 1
        ends   = np.where(transitions == -1)[0] + 1
        if above[0]:
            starts = np.concatenate([[0], starts])
        if above[-1]:
            ends = np.concatenate([ends, [len(x)]])

        for s, e in zip(starts, ends):
            region_x = x[s:e]
            region_d2 = d2[s:e]
            width = region_x[-1] - region_x[0]
            centre = float(region_x[np.argmax(np.abs(region_d2))])

            # Oscillatory defect: cluster near origin with sign changes
            sign_changes = np.sum(np.diff(np.sign(region_d2)) != 0)
            if abs(centre) < eps_osc and sign_changes > 2:
                detected.append(
                    OscillatoryDefect(eps=max(width, 0.05),
                                      beta=float(np.max(np.abs(region_d2))) * 0.1))
            # Step defect: narrow, single peak
            elif width < 0.25:
                detected.append(
                    StepDefect(x0=centre,
                               alpha=float(np.sign(np.mean(region_d2)))))
            # Conditional defect: wide interval
            else:
                amp = float(np.mean(region_d2)) * 0.05
                detected.append(
                    ConditionalDefect(
                        intervals=[(float(region_x[0]), float(region_x[-1]))],
                        amplitudes=[amp]))

        return detected

    def __repr__(self) -> str:
        return f"DefectAlgebra(N={len(self.x)}, generators=[Ŝ, Ô, Ĉ])"
