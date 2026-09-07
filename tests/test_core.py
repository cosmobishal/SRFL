import numpy as np
import pytest

from srfl import (
    DefectCandidate,
    DefectDetector,
    DefectRegistry,
    GaussianKernel,
    Grid1D,
    SRFLConfig,
    SRFLRegressor,
    SRFLSolver,
    Swarm,
)


def grid():
    return np.linspace(-np.pi, np.pi, 256)


def step_target(x):
    return (x >= 0).astype(float)


def test_grid_validation():
    g = Grid1D(grid())
    assert g.n == 256
    assert g.dx > 0
    with pytest.raises(ValueError):
        Grid1D(np.array([0, 1, 3, 4, 5, 6, 7, 8], dtype=float))


def test_kernel_preserves_constant_and_shape():
    x = grid()
    k = GaussianKernel(Grid1D(x), 0.2)
    y = k.convolve(np.full_like(x, 3.0))
    assert y.shape == x.shape
    assert np.allclose(y, 3.0, atol=1e-10)


def test_kernel_smooths_high_frequency_signal():
    x = grid()
    k = GaussianKernel(Grid1D(x), 0.2)
    y = k.convolve(np.sin(40 * x))
    assert np.std(y) < np.std(np.sin(40 * x))


def test_detector_returns_structured_candidates():
    x = grid()
    f = step_target(x)
    field = GaussianKernel(Grid1D(x), 0.3).convolve(f)
    detector = DefectDetector(sensitivity=3.0, max_candidates=4)
    candidates = detector.detect(x, field, f - field, 0.2)
    assert candidates
    assert all(c.kind in {"step", "oscillatory", "piecewise"} for c in candidates)
    assert all(np.isfinite(c.score) for c in candidates)


def test_defect_registry_shape():
    x = grid()
    reg = DefectRegistry()
    for kind in reg.names():
        c = DefectCandidate(kind, 0.0, 0.15, 5.0, 1.0, 0.5)
        out = reg.evaluate(x, c)
        assert out.shape == x.shape
        assert np.all(np.isfinite(out))


def test_swarm_spawn_and_merge():
    x = grid()
    swarm = Swarm(x, n_init=0, merge_distance=0.2)
    c1 = DefectCandidate("step", 0.0, 0.1, 8.0, 1.0, 0.7)
    c2 = DefectCandidate("step", 0.01, 0.1, 7.0, 1.0, 0.5)
    swarm.update([c1], 0.2, 0)
    assert swarm.count() == 1
    swarm.update([c2], 0.15, 1)
    assert swarm.count() == 1
    assert swarm.event_summary()["spawn"] >= 1


def test_solver_improves_step_without_periodic_wrap_artifact():
    x = grid()
    y = step_target(x)
    result = SRFLSolver(SRFLConfig(scales=28, lam_max=0.8, lam_min=0.03,
                                   dt=0.4, n_agents=4, seed=123)).fit(x, y)
    assert result.final_error < result.initial_error
    assert np.max(np.abs(result.final_field)) <= 1.15 + 1e-8
    assert len(result.events) >= 0
    assert np.isfinite(result.final_error)
    # A reflective finite-domain solver should keep the two ends distinct for this target.
    assert abs(result.final_field[0] - result.final_field[-1]) > 0.2


def test_regressor_predict_and_score():
    x = grid()
    y = np.sin(2 * x)
    model = SRFLRegressor(scales=20, lam_max=0.6, lam_min=0.04, dt=0.35, n_agents=3)
    model.fit(x, y)
    pred = model.predict(np.array([-1.0, 0.0, 1.0]))
    assert pred.shape == (3,)
    assert np.isfinite(model.score(x, y))


def test_config_rejects_bad_scale_range():
    with pytest.raises(ValueError):
        SRFLConfig(lam_max=0.1, lam_min=0.2).validate()
