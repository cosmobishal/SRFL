"""
srfl.action
===========
Action functional  𝒜 = 𝒜_data + 𝒜_scale + 𝒜_sym + 𝒜_cplx

This is the SRFL replacement for empirical risk / loss functions.
No gradient is taken — the action is evaluated as a diagnostic and
used to monitor convergence.

    𝒜_data  = ∫∫ (Φ(x,λ) − f(x))² · w_data(λ) dx dλ
    𝒜_scale = ∫∫∫ (Φ(x,λ₁) − Π(λ₁→λ₂)Φ(x))² · w(λ₁,λ₂) dx dλ₁ dλ₂
    𝒜_sym   = ∫∫ (Φ(x,λ) − ℛ[Φ](x,λ))² · χ_sym(x) dx dλ
    𝒜_cplx  = ∫ c(λ) · Σᵢ ‖Dᵢ‖²_𝒟 dλ
"""

import numpy as np
from typing import List, Optional, Callable


class ActionFunctional:
    """
    Four-term SRFL action functional.

    Parameters
    ----------
    x          : np.ndarray  — spatial grid
    lam_sched  : np.ndarray  — scale schedule λ(s)
    target     : np.ndarray  — target function f(x)
    beta       : float       — complexity exponent: c(λ) = λ^β
    symmetry_op: callable    — symmetry operator ℛ (optional)
    """

    def __init__(
        self,
        x:           np.ndarray,
        lam_sched:   np.ndarray,
        target:      np.ndarray,
        beta:        float = 1.0,
        symmetry_op: Optional[Callable] = None,
    ):
        self.x           = x
        self.lam_sched   = lam_sched
        self.target      = target
        self.beta        = beta
        self.symmetry_op = symmetry_op
        self.dx          = float(x[1] - x[0])
        self.dlam        = float(np.abs(np.mean(np.diff(lam_sched))))

    # ------------------------------------------------------------------ #

    def _w_data(self, lam: float) -> float:
        """Data weight: w_data(λ) = λ⁻¹ · e^{-λ}.  Fine scales weighted more."""
        return np.exp(-lam) / (lam + 1e-12)

    def _convolve_1d(self, phi: np.ndarray, lam: float) -> np.ndarray:
        """Gaussian convolution at scale λ via FFT."""
        N     = len(phi)
        freqs = np.fft.rfftfreq(N, d=self.dx)
        K_hat = np.exp(-2.0 * np.pi**2 * freqs**2 * lam**2)
        return np.fft.irfft(np.fft.rfft(phi) * K_hat, n=N)

    def _project(self, phi: np.ndarray, lam1: float, lam2: float) -> np.ndarray:
        """
        Scale projection  Π(λ₁→λ₂)[Φ]:
            convolve with K_{√(λ₂²−λ₁²)} if λ₂ > λ₁.
        """
        if lam2 <= lam1:
            return phi  # deconvolution not implemented here
        dlam = np.sqrt(max(lam2**2 - lam1**2, 0.0))
        if dlam < 1e-10:
            return phi
        return self._convolve_1d(phi, dlam)

    # ------------------------------------------------------------------ #

    def A_data(self, fields: List[np.ndarray]) -> float:
        """
        Data fidelity term:
            𝒜_data = ∫∫ (Φ(x,λ) − f(x))² · w_data(λ) dx dλ
        """
        total = 0.0
        for k, (phi, lam) in enumerate(zip(fields, self.lam_sched)):
            integrand = np.mean((phi - self.target)**2)
            total    += integrand * self._w_data(lam)
        return float(total * self.dlam * self.dx)

    def A_scale(self, fields: List[np.ndarray],
                stride: int = 5) -> float:
        """
        Scale consistency term:
            𝒜_scale = ∫∫∫ (Φ(x,λ₁) − Π(λ₁→λ₂)Φ(x))² dx dλ₁ dλ₂

        Uses strided λ-pairs for efficiency.
        """
        total = 0.0
        n = len(fields)
        count = 0
        for i in range(0, n - stride, stride):
            j     = min(i + stride, n - 1)
            phi_i = fields[i]
            phi_j = fields[j]
            lam_i = self.lam_sched[i]
            lam_j = self.lam_sched[j]
            proj  = self._project(phi_i, lam_i, lam_j)
            total += float(np.mean((phi_j - proj)**2))
            count += 1
        return float(total / max(count, 1))

    def A_sym(self, fields: List[np.ndarray]) -> float:
        """
        Symmetry term:
            𝒜_sym = ∫∫ (Φ − ℛ[Φ])² · χ_sym dx dλ
        If no symmetry operator provided, returns 0.
        """
        if self.symmetry_op is None:
            return 0.0
        total = 0.0
        for phi, lam in zip(fields, self.lam_sched):
            R_phi     = self.symmetry_op(phi)
            total    += float(np.mean((phi - R_phi)**2))
        return float(total * self.dlam * self.dx)

    def A_cplx(self, defect_norms: List[float]) -> float:
        """
        Complexity term:
            𝒜_cplx = ∫ c(λ) · Σᵢ ‖Dᵢ‖²_𝒟 dλ

        Parameters
        ----------
        defect_norms : list of total defect norms at each scale step.
        """
        total = 0.0
        for k, (norm, lam) in enumerate(zip(defect_norms, self.lam_sched)):
            c_lam  = lam**self.beta
            total += c_lam * norm**2
        return float(total * self.dlam)

    def total(self, fields: List[np.ndarray],
              defect_norms: Optional[List[float]] = None,
              stride: int = 5) -> dict:
        """
        Compute all four action terms.

        Returns
        -------
        dict with keys: data, scale, sym, cplx, total
        """
        if defect_norms is None:
            defect_norms = [0.0] * len(fields)

        A_d = self.A_data(fields)
        A_s = self.A_scale(fields, stride=stride)
        A_y = self.A_sym(fields)
        A_c = self.A_cplx(defect_norms)
        A_t = A_d + A_s + A_y + A_c

        return {
            "data"       : A_d,
            "scale"      : A_s,
            "symmetry"   : A_y,
            "complexity" : A_c,
            "total"      : A_t,
        }

    def __repr__(self) -> str:
        return (f"ActionFunctional(N={len(self.x)}, "
                f"S={len(self.lam_sched)}, β={self.beta})")
