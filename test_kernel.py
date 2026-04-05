"""Tests for srfl.kernel"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
try:
    import pytest
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    import pytest_shim as pytest
from srfl import SRFLKernel


N  = 256
x  = np.linspace(-np.pi, np.pi, N)
dx = float(x[1] - x[0])


class TestSRFLKernel:

    def test_init_positive_lambda(self):
        K = SRFLKernel(x, lam=0.5)
        assert K.lam == 0.5

    def test_init_negative_lambda_raises(self):
        with pytest.raises(ValueError):
            SRFLKernel(x, lam=-0.1)

    def test_init_zero_lambda_raises(self):
        with pytest.raises(ValueError):
            SRFLKernel(x, lam=0.0)

    def test_convolve_preserves_shape(self):
        K   = SRFLKernel(x, lam=0.3)
        phi = np.random.randn(N)
        out = K.convolve(phi)
        assert out.shape == (N,)

    def test_convolve_smooths(self):
        """Convolution must reduce high-frequency power."""
        K   = SRFLKernel(x, lam=0.5)
        phi = np.sin(50 * x)       # high-frequency input
        out = K.convolve(phi)
        assert np.std(out) < np.std(phi)

    def test_convolve_constant_invariant(self):
        """K * c = c for constant field (normalisation)."""
        K   = SRFLKernel(x, lam=0.4)
        phi = np.ones(N) * 3.14
        out = K.convolve(phi)
        assert np.allclose(out, phi, atol=1e-4)

    def test_approximate_identity(self):
        """As λ→0, K*φ → φ in L²."""
        K   = SRFLKernel(x, lam=1e-3)
        phi = np.sin(2 * x)
        out = K.convolve(phi)
        assert np.sqrt(np.mean((out - phi)**2)) < 0.01

    def test_semigroup(self):
        """K_λ * K_μ = K_{√(λ²+μ²)}."""
        lam, mu  = 0.3, 0.4
        K1 = SRFLKernel(x, lam=lam)
        K2 = SRFLKernel(x, lam=mu)
        K3 = SRFLKernel(x, lam=np.sqrt(lam**2 + mu**2))
        phi = np.sin(x) + 0.5 * np.cos(3*x)
        chain  = K2.convolve(K1.convolve(phi))
        direct = K3.convolve(phi)
        assert np.sqrt(np.mean((chain - direct)**2)) < 1e-6

    def test_fwhm(self):
        """FWHM = 2√(2ln2)·λ."""
        lam = 0.6
        K   = SRFLKernel(x, lam=lam)
        expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * lam
        assert abs(K.fwhm() - expected) < 1e-12

    def test_matrix_shape(self):
        K  = SRFLKernel(x, lam=0.5)
        M  = K.matrix(subsample=4)
        ns = N // 4
        assert M.shape == (ns, ns)

    def test_agent_kernel_self_is_one(self):
        K = SRFLKernel(x, lam=0.3)
        v = K.agent_kernel(0.5, 0.5, lam_i=0.3, lam_j=0.3)
        assert abs(v - 1.0) < 1e-10

    def test_agent_kernel_decreases_with_distance(self):
        K  = SRFLKernel(x, lam=0.5)
        v1 = K.agent_kernel(0.0, 0.1)
        v2 = K.agent_kernel(0.0, 1.0)
        assert v1 > v2

    def test_update_lambda(self):
        K1 = SRFLKernel(x, lam=0.5)
        K2 = K1.update_lambda(0.1)
        assert K2.lam == 0.1
        assert K1.lam == 0.5   # original unchanged

    def test_repr(self):
        K = SRFLKernel(x, lam=0.25)
        r = repr(K)
        assert "SRFLKernel" in r and "0.25" in r
