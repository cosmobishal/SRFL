"""Tests for srfl.multiscale -- ScaleProjection"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srfl import ScaleProjection, SRFLField

x         = np.linspace(-np.pi, np.pi, 256)
lam_sched = np.logspace(0, -1.5, 30)
target    = np.where(x >= 0, 1.0, 0.0).astype(float)


@pytest.fixture(scope="module")
def proj():
    return ScaleProjection(x)


@pytest.fixture(scope="module")
def phi():
    return np.sin(x) + 0.3 * np.cos(3 * x)


@pytest.fixture(scope="module")
def fields():
    engine = SRFLField(x, target, lam_sched, dt=0.25)
    fs, _  = engine.run()
    return fs


class TestScaleProjection:

    def test_project_same_scale_identity(self, proj, phi):
        """Pi(lam -> lam)[Phi] = Phi."""
        out = proj.project(phi, 0.5, 0.5)
        assert np.allclose(out, phi)

    def test_project_output_shape(self, proj, phi):
        assert proj.project(phi, 0.3, 0.7).shape == (len(x),)

    def test_project_smooths(self, proj, phi):
        """Projecting to a coarser scale reduces variance."""
        coarse = proj.project(phi, 0.1, 0.9)
        assert np.var(coarse) <= np.var(phi) + 1e-8

    def test_project_reverse_raises(self, proj, phi):
        """Pi(lam1 -> lam2) with lam2 < lam1 must raise ValueError."""
        with pytest.raises(ValueError):
            proj.project(phi, 0.8, 0.2)

    def test_semigroup_property(self, proj, phi):
        """Pi(lam2->lam3) o Pi(lam1->lam2) == Pi(lam1->lam3) up to tolerance."""
        lam1, lam2, lam3 = 0.2, 0.5, 0.8
        err, ok = proj.verify_semigroup(phi, lam1, lam2, lam3, tol=1e-5)
        assert ok, f"Semigroup error {err:.2e} exceeds tolerance"

    def test_semigroup_error_is_small(self, proj, phi):
        err, _ = proj.verify_semigroup(phi, 0.1, 0.4, 0.7, tol=1e-4)
        assert err < 1e-4

    def test_consistency_profile_shape(self, proj, fields):
        profile = proj.consistency_profile(fields, lam_sched, stride=5)
        assert profile.shape == (len(fields),)

    def test_consistency_profile_non_negative(self, proj, fields):
        profile = proj.consistency_profile(fields, lam_sched, stride=5)
        assert np.all(profile >= 0.0)

    def test_l2_error_profile_shape(self, proj, fields):
        assert proj.l2_error_profile(fields, target).shape == (len(fields),)

    def test_l2_error_profile_decreasing_trend(self, proj, fields):
        """L2 error should fall from first to last step."""
        profile = proj.l2_error_profile(fields, target)
        assert profile[-1] < profile[0]

    def test_repr(self, proj):
        assert "ScaleProjection" in repr(proj)
