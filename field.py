"""
srfl.field
==========
SRFL field evolution engine.

Integrates the core SRFL equation:

    ∂Φ/∂s(x,s) = ∫ K(x,x',λ(s)) [f(x') − Φ(x',s)] dx'  +  𝒮[Φ](x,s)

from s=0 (coarse scale λ₀) to s=S (fine scale λ_S → 0).

Key classes
-----------
SingularityGenerator  : 𝒮[Φ] operator
SRFLField             : Full field evolution engine
"""

import numpy as np
from typing import Callable, List, Optional, Tuple

from .kernel import SRFLKernel


# ─────────────────────────────────────────────────────────────────────────────
class SingularityGenerator:
    """
    Singularity generator  𝒮[Φ](x).

    𝒮[Φ](x) = ε · sgn(∂²Φ/∂x²) · ReLU(|∂²Φ/∂x²| − κ)

    Active where local curvature |∂²Φ| > κ.
    Injects defect seeds aligned with the sign of the curvature.

    Parameters
    ----------
    kappa : float  — instability threshold κ > 0
    eps   : float  — injection amplitude ε > 0
    """

    def __init__(self, kappa: float = 1.8, eps: float = 0.07):
        self.kappa = kappa
        self.eps   = eps

    def __call__(self, phi: np.ndarray, dx: float) -> np.ndarray:
        """
        Evaluate 𝒮[Φ](x).

        Parameters
        ----------
        phi : np.ndarray  — current field Φ(·, s)
        dx  : float       — grid spacing

        Returns
        -------
        np.ndarray  — defect injection field 𝒮[Φ]
        """
        d2   = np.gradient(np.gradient(phi, dx), dx)
        mag  = np.abs(d2)
        D    = np.zeros_like(phi)
        mask = mag > self.kappa
        D[mask] = (self.eps
                   * np.sign(d2[mask])
                   * np.tanh(mag[mask] - self.kappa))
        return D

    def __repr__(self) -> str:
        return f"SingularityGenerator(κ={self.kappa}, ε={self.eps})"


# ─────────────────────────────────────────────────────────────────────────────
class SRFLField:
    """
    Full SRFL field evolution engine.

    Integrates:
        ∂Φ/∂s = K_λ * (f − Φ)  +  α_S · 𝒮[Φ]

    via forward Euler in pseudo-time s.

    Parameters
    ----------
    x          : np.ndarray        — spatial grid, shape (N,)
    target     : np.ndarray        — target function f(x), shape (N,)
    lam_sched  : np.ndarray        — λ schedule (decreasing), shape (S,)
    dt         : float             — pseudo-time step Δs
    alpha_S    : float             — singularity weight
    kappa      : float             — instability threshold κ
    eps_S      : float             — singularity amplitude ε
    clip       : float             — field clip bound (prevents blow-up)

    Examples
    --------
    >>> import numpy as np
    >>> from srfl import SRFLField
    >>> x   = np.linspace(-np.pi, np.pi, 512)
    >>> f   = np.where(x >= 0, 1.0, 0.0)   # step function
    >>> lam = np.logspace(0, -1.85, 70)
    >>> engine = SRFLField(x, f, lam)
    >>> fields, errors = engine.run()
    """

    def __init__(
        self,
        x:         np.ndarray,
        target:    np.ndarray,
        lam_sched: np.ndarray,
        dt:        float = 0.28,
        alpha_S:   float = 0.12,
        kappa:     float = 1.8,
        eps_S:     float = 0.07,
        clip:      float = 2.8,
    ):
        self.x          = x
        self.target     = target
        self.lam_sched  = lam_sched
        self.dt         = dt
        self.alpha_S    = alpha_S
        self.clip       = clip
        self.dx         = float(x[1] - x[0])
        self.N          = len(x)
        self.S_gen      = SingularityGenerator(kappa=kappa, eps=eps_S)

        # Validate
        if len(target) != self.N:
            raise ValueError("target must have same length as x")
        if not np.all(np.diff(lam_sched) <= 0):
            raise ValueError("lam_sched must be non-increasing")

    # ------------------------------------------------------------------ #

    def _G(self, phi: np.ndarray) -> np.ndarray:
        """Functional response  𝒢[Φ] = tanh(Φ)."""
        return np.tanh(phi)

    def _convolve(self, phi: np.ndarray, lam: float) -> np.ndarray:
        """Gaussian convolution via FFT."""
        freqs = np.fft.rfftfreq(self.N, d=self.dx)
        K_hat = np.exp(-2.0 * np.pi**2 * freqs**2 * lam**2)
        return np.fft.irfft(np.fft.rfft(phi) * K_hat, n=self.N)

    def _step(self, phi: np.ndarray, lam: float) -> np.ndarray:
        """Single Euler step at scale λ."""
        driven = self._convolve(self.target - phi, lam)
        sing   = self.S_gen(phi, self.dx)
        phi_new = phi + self.dt * (driven + self.alpha_S * sing)
        return np.clip(phi_new, -self.clip, self.clip)

    # ------------------------------------------------------------------ #

    def run(self, verbose: bool = False
            ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Run the full SRFL evolution.

        Returns
        -------
        fields : list of np.ndarray
            Field snapshots Φ(·, s_k) at each scale step k.
        errors : list of float
            L² error ‖Φ(·,s_k) − f‖_{L²} at each step.
        """
        # Initialise at coarsest scale
        phi = self._convolve(self.target, self.lam_sched[0])
        fields = [phi.copy()]
        errors = [float(np.sqrt(np.mean((phi - self.target)**2)))]

        for k in range(1, len(self.lam_sched)):
            lam = self.lam_sched[k]
            phi = self._step(phi, lam)
            fields.append(phi.copy())
            err = float(np.sqrt(np.mean((phi - self.target)**2)))
            errors.append(err)
            if verbose and k % 10 == 0:
                print(f"  step {k:3d}/{len(self.lam_sched)-1}"
                      f"  λ={lam:.4f}  L²err={err:.4f}")

        return fields, errors

    def final_field(self) -> np.ndarray:
        """Return only the final field Φ(·, S)."""
        fields, _ = self.run()
        return fields[-1]

    def convergence_rate(self, errors: List[float]) -> float:
        """
        Estimate the convergence rate r from errors ≈ C · λ^r
        by log-linear regression of log(error) vs log(λ).
        """
        lams = self.lam_sched
        valid = np.array(errors) > 1e-10
        if valid.sum() < 4:
            return float("nan")
        log_lam = np.log(lams[valid])
        log_err = np.log(np.array(errors)[valid])
        r = float(np.polyfit(log_lam, log_err, 1)[0])
        return r

    def __repr__(self) -> str:
        return (f"SRFLField(N={self.N}, "
                f"S={len(self.lam_sched)}, "
                f"λ∈[{self.lam_sched[-1]:.3f},{self.lam_sched[0]:.3f}])")
