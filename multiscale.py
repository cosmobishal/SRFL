"""
srfl.multiscale
===============
Scale projection  Π(λ₁ → λ₂)  and scale-consistency diagnostics.

Π(λ₁ → λ₂)[Φ](x) = ∫ K(x, x', √(λ₂²−λ₁²)) Φ(x', λ₁) dx'   (λ₂ > λ₁)

Semigroup property:
    Π(λ₂→λ₃) ∘ Π(λ₁→λ₂) = Π(λ₁→λ₃)  ∀ λ₁ < λ₂ < λ₃
"""

import numpy as np
from typing import List, Tuple


class ScaleProjection:
    """
    Scale projection operator  Π(λ₁ → λ₂).

    Parameters
    ----------
    x  : np.ndarray  — spatial grid
    """

    def __init__(self, x: np.ndarray):
        self.x  = x
        self.N  = len(x)
        self.dx = float(x[1] - x[0])

    def _convolve(self, phi: np.ndarray, lam: float) -> np.ndarray:
        """FFT-based Gaussian convolution at scale lam."""
        freqs = np.fft.rfftfreq(self.N, d=self.dx)
        K_hat = np.exp(-2.0 * np.pi**2 * freqs**2 * lam**2)
        return np.fft.irfft(np.fft.rfft(phi) * K_hat, n=self.N)

    def project(self, phi: np.ndarray,
                lam1: float, lam2: float) -> np.ndarray:
        """
        Project field from scale λ₁ to scale λ₂.

        Parameters
        ----------
        phi   : np.ndarray  — field Φ(·, λ₁)
        lam1  : float       — source scale λ₁
        lam2  : float       — target scale λ₂  (must be ≥ λ₁)

        Returns
        -------
        np.ndarray  — Π(λ₁→λ₂)[Φ]
        """
        if abs(lam2 - lam1) < 1e-12:
            return phi.copy()
        if lam2 < lam1:
            raise ValueError(
                f"Forward projection requires λ₂ ≥ λ₁, got λ₁={lam1}, λ₂={lam2}. "
                "Deconvolution (λ₂ < λ₁) is not implemented.")
        delta_lam = np.sqrt(lam2**2 - lam1**2)
        return self._convolve(phi, delta_lam)

    def verify_semigroup(self, phi: np.ndarray,
                         lam1: float, lam2: float, lam3: float,
                         tol: float = 1e-6) -> Tuple[float, bool]:
        """
        Verify the semigroup property:
            ‖Π(λ₂→λ₃)∘Π(λ₁→λ₂)[Φ] − Π(λ₁→λ₃)[Φ]‖_{L²}  ≤  tol

        Returns
        -------
        error : float   — L² discrepancy
        ok    : bool    — True if within tolerance
        """
        chain  = self.project(self.project(phi, lam1, lam2), lam2, lam3)
        direct = self.project(phi, lam1, lam3)
        error  = float(np.sqrt(np.mean((chain - direct)**2)))
        return error, error <= tol

    def consistency_profile(self, fields: List[np.ndarray],
                            lam_sched: np.ndarray,
                            stride: int = 5) -> np.ndarray:
        """
        Compute scale-consistency error at each scale step.

        Since lam_sched is *decreasing* (coarse → fine), the pair
        (k-stride, k) has lam_prev > lam_curr.  The consistency check is:

            err(k) = ‖Φ(·,λ_coarse) − Π(λ_fine → λ_coarse)[Φ(·,λ_fine)]‖_{L²}

        i.e., blurring the fine field up to coarse resolution should
        recover the coarse field.

        Returns
        -------
        np.ndarray of shape (len(fields),) — zero at first stride steps
        """
        n   = len(fields)
        err = np.zeros(n)
        for k in range(stride, n):
            phi_coarse = fields[k - stride]        # earlier = coarser λ
            phi_fine   = fields[k]                 # later   = finer   λ
            lam_coarse = float(lam_sched[k - stride])
            lam_fine   = float(lam_sched[k])
            # Project fine → coarse (lam2 > lam1 ✓)
            proj   = self.project(phi_fine, lam_fine, lam_coarse)
            err[k] = float(np.sqrt(np.mean((phi_coarse - proj)**2)))
        return err

    def l2_error_profile(self, fields: List[np.ndarray],
                         target: np.ndarray) -> np.ndarray:
        """L² error ‖Φ(·,λ_k) − f‖ at each scale step."""
        return np.array([np.sqrt(np.mean((phi - target)**2)) for phi in fields])

    def __repr__(self) -> str:
        return f"ScaleProjection(N={self.N})"
