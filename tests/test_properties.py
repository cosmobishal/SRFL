import numpy as np

from srfl import GaussianKernel, Grid1D, SRFLSolver, SRFLConfig


def test_wrap_kernel_gaussian_semigroup_for_periodic_signal():
    x = np.linspace(-np.pi, np.pi, 256, endpoint=False)
    g = Grid1D(x)
    phi = np.sin(3 * x) + 0.3 * np.cos(7 * x)
    a, b = 0.17, 0.23
    chained = GaussianKernel(g, b, "wrap").convolve(
        GaussianKernel(g, a, "wrap").convolve(phi)
    )
    direct = GaussianKernel(g, np.sqrt(a * a + b * b), "wrap").convolve(phi)
    assert np.sqrt(np.mean((chained - direct) ** 2)) < 1e-6


def test_solver_determinism():
    x = np.linspace(-2, 2, 192)
    y = np.exp(-3 * x * x) + 0.2 * np.sin(7 * x)
    cfg = SRFLConfig(scales=18, lam_max=0.5, lam_min=0.05, dt=0.3, n_agents=4, seed=99)
    a = SRFLSolver(cfg).fit(x, y)
    b = SRFLSolver(cfg).fit(x, y)
    assert np.allclose(a.final_field, b.final_field)
    assert np.allclose(a.errors, b.errors)
