"""Tests for srfl.multiscale — ScaleProjection"""
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
from srfl import ScaleProjection, SRFLField

x         = np.linspace(-np.pi, np.pi, 256)
lam_sched = np.logspace(0, -1.5, 30)
target    = np.where(x >= 0, 1.0, 0.0).astype(float)


class TestScaleProjection:

    def setup_method(self):
        self.proj = ScaleProjection(x)
        self.phi  = np.sin(x) + 0.3 * np.cos(3 * x)

    def test_project_same_scale_identity(self):
        """Π(λ→λ)[Φ] = Φ."""
        out = self.proj.project(self.phi, 0.5, 0.5)
        assert np.allclose(out, self.phi)

    def test_project_output_shape(self):
        out = self.proj.project(self.phi, 0.3, 0.7)
        assert out.shape == (len(x),)

    def test_project_smooths(self):
        """Projecting to coarser scale must reduce variance."""
        fine   = self.phi.copy()
        coarse = self.proj.project(fine, 0.1, 0.9)
        assert np.var(coarse) <= np.var(fine) + 1e-8

    def test_project_reverse_raises(self):
        """Π(λ₁→λ₂) with λ₂ < λ₁ should raise ValueError."""
        with pytest.raises(ValueError):
            self.proj.project(self.phi, 0.8, 0.2)

    def test_semigroup_property(self):
        """Π(λ₂→λ₃)∘Π(λ₁→λ₂) == Π(λ₁→λ₃) up to numerical tolerance."""
        lam1, lam2, lam3 = 0.2, 0.5, 0.8
        err, ok = self.proj.verify_semigroup(self.phi, lam1, lam2, lam3,
                                              tol=1e-5)
        assert ok, f"Semigroup error {err:.2e} exceeds tolerance"

    def test_semigroup_error_is_small(self):
        err, _ = self.proj.verify_semigroup(self.phi, 0.1, 0.4, 0.7, tol=1e-4)
        assert err < 1e-4

    def test_consistency_profile_shape(self):
        engine = SRFLField(x, target, lam_sched, dt=0.25)
        fields, _ = engine.run()
        profile = self.proj.consistency_profile(fields, lam_sched, stride=5)
        assert profile.shape == (len(fields),)

    def test_consistency_profile_non_negative(self):
        engine = SRFLField(x, target, lam_sched, dt=0.25)
        fields, _ = engine.run()
        profile = self.proj.consistency_profile(fields, lam_sched, stride=5)
        assert np.all(profile >= 0.0)

    def test_l2_error_profile_shape(self):
        engine = SRFLField(x, target, lam_sched, dt=0.25)
        fields, _ = engine.run()
        profile = self.proj.l2_error_profile(fields, target)
        assert profile.shape == (len(fields),)

    def test_l2_error_profile_decreasing_trend(self):
        """L² error should trend downward from first to last step."""
        engine = SRFLField(x, target, lam_sched, dt=0.25)
        fields, _ = engine.run()
        profile = self.proj.l2_error_profile(fields, target)
        assert profile[-1] < profile[0]

    def test_repr(self):
        assert "ScaleProjection" in repr(self.proj)
