"""
srfl.kernel
===========
Non-local Gaussian kernel  K(x, x', λ).

Definition
----------
    K(x, x', λ) = Z_λ⁻¹ · exp(-(x-x')² / (2λ²))
    Z_λ = √(2πλ²)

In Fourier space:
    K̂(ξ, λ) = exp(-2π²ξ²λ²)

which enables O(N log N) convolution via FFT.
"""

import numpy as np


class SRFLKernel:
    """
    Gaussian renormalization kernel operating on a 1-D grid.

    Parameters
    ----------
    x   : np.ndarray, shape (N,)
        Uniform spatial grid on Ω ⊂ ℝ.
    lam : float
        Scale parameter λ > 0.

    Examples
    --------
    >>> import numpy as np
    >>> from srfl import SRFLKernel
    >>> x = np.linspace(-np.pi, np.pi, 512)
    >>> K = SRFLKernel(x, lam=0.5)
    >>> phi_smooth = K.convolve(phi)
    """

    def __init__(self, x: np.ndarray, lam: float):
        if lam <= 0:
            raise ValueError(f"Scale λ must be positive, got {lam}")
        self.x   = x
        self.lam = lam
        self.N   = len(x)
        self.dx  = float(x[1] - x[0])
        # Pre-compute FFT frequency filter
        freqs        = np.fft.rfftfreq(self.N, d=self.dx)
        self._K_hat  = np.exp(-2.0 * np.pi**2 * freqs**2 * lam**2)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def convolve(self, phi: np.ndarray) -> np.ndarray:
        """
        Compute the non-local kernel action:
            (K_λ * φ)(x) = ∫ K(x, x', λ) φ(x') dx'

        Parameters
        ----------
        phi : np.ndarray, shape (N,)

        Returns
        -------
        np.ndarray, shape (N,)
        """
        return np.fft.irfft(np.fft.rfft(phi) * self._K_hat, n=self.N)

    def matrix(self, subsample: int = 1) -> np.ndarray:
        """
        Return the dense kernel matrix K[i, j] = K(xᵢ, xⱼ, λ).
        Subsampled for memory efficiency.

        Parameters
        ----------
        subsample : int
            Take every `subsample`-th grid point.

        Returns
        -------
        np.ndarray, shape (M, M)  where M = N // subsample
        """
        xs = self.x[::subsample]
        diff = xs[:, None] - xs[None, :]
        K = np.exp(-diff**2 / (2 * self.lam**2))
        K /= K.sum(axis=1, keepdims=True) * self.dx
        return K

    def fwhm(self) -> float:
        """Full-width at half-maximum: 2√(2 ln 2) · λ."""
        return 2.0 * np.sqrt(2.0 * np.log(2.0)) * self.lam

    def agent_kernel(self, xi: float, xj: float,
                     lam_i: float = None, lam_j: float = None,
                     sigma_lam: float = 0.1) -> float:
        """
        Inter-agent interaction kernel:
            Kᵢⱼ = exp(-(xᵢ-xⱼ)²/(2λ²)) · exp(-(λᵢ-λⱼ)²/(2σ_λ²))

        Parameters
        ----------
        xi, xj         : agent spatial positions
        lam_i, lam_j   : agent scale parameters (optional)
        sigma_lam       : scale bandwidth σ_λ

        Returns
        -------
        float ∈ (0, 1]
        """
        K_space = float(np.exp(-(xi - xj)**2 / (2 * self.lam**2)))
        if lam_i is not None and lam_j is not None:
            K_scale = float(np.exp(-(lam_i - lam_j)**2 / (2 * sigma_lam**2)))
        else:
            K_scale = 1.0
        return K_space * K_scale

    def update_lambda(self, lam: float) -> "SRFLKernel":
        """Return a new kernel with updated λ (immutable update)."""
        return SRFLKernel(self.x, lam)

    # ------------------------------------------------------------------ #
    #  Dunder                                                              #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return f"SRFLKernel(N={self.N}, λ={self.lam:.4f}, FWHM={self.fwhm():.4f})"
