"""Tests for srfl.field -- SRFLField and SingularityGenerator"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srfl import SRFLField
from srfl.field import SingularityGenerator

x         = np.linspace(-np.pi, np.pi, 256)
dx        = float(x[1] - x[0])
lam_sched = np.logspace(0, -1.5, 30)
target    = np.where(x >= 0, 1.0, 0.0).astype(float)


class TestSingularityGenerator:

    def test_output_shape(self):
        S   = SingularityGenerator(kappa=1.8, eps=0.07)
        out = S(np.sin(x), dx)
        assert out.shape == (len(x),)

    def test_zero_for_smooth_field(self):
        """Low-amplitude smooth field: curvature < kappa, S = 0."""
        S   = SingularityGenerator(kappa=10.0, eps=0.07)
        out = S(0.01 * np.sin(x), dx)
        assert np.allclose(out, 0.0)

    def test_nonzero_for_steep_field(self):
        """Steep step: large curvature fires the singularity generator."""
        S   = SingularityGenerator(kappa=0.5, eps=0.1)
        phi = np.where(x >= 0, 1.0, 0.0).astype(float)
        out = S(phi, dx)
        assert np.any(out != 0.0)

    def test_bounded_output(self):
        """Output is always finite (tanh keeps it bounded)."""
        S   = SingularityGenerator(kappa=0.1, eps=0.5)
        phi = np.random.randn(len(x)) * 10
        out = S(phi, dx)
        assert np.all(np.isfinite(out))

    def test_sign_alignment(self):
        """Output sign must match sign of d2Phi where active."""
        S   = SingularityGenerator(kappa=0.3, eps=0.1)
        phi = np.where(x >= 0, 1.0, 0.0).astype(float)
        d2  = np.gradient(np.gradient(phi, dx), dx)
        out = S(phi, dx)
        active = np.abs(d2) > 0.3
        if active.any():
            assert np.all(np.sign(out[active]) == np.sign(d2[active]))


class TestSRFLField:

    def test_init_validates_target_length(self):
        with pytest.raises(ValueError):
            SRFLField(x, np.ones(100), lam_sched)

    def test_init_validates_lam_sched_decreasing(self):
        bad = np.linspace(0.01, 1.0, 20)   # increasing -- invalid
        with pytest.raises(ValueError):
            SRFLField(x, target, bad)

    def test_run_returns_correct_lengths(self):
        fields, errors = SRFLField(x, target, lam_sched).run()
        assert len(fields) == len(lam_sched)
        assert len(errors) == len(lam_sched)

    def test_fields_have_correct_shape(self):
        fields, _ = SRFLField(x, target, lam_sched).run()
        for f in fields:
            assert f.shape == (len(x),)

    def test_errors_non_negative(self):
        _, errors = SRFLField(x, target, lam_sched).run()
        assert all(e >= 0.0 for e in errors)

    def test_error_converges(self):
        """Final L2 error must be strictly less than initial."""
        _, errors = SRFLField(x, target, lam_sched).run()
        assert errors[-1] < errors[0]

    def test_field_clipped(self):
        """All field values must lie within [-clip, clip]."""
        clip      = 2.8
        fields, _ = SRFLField(x, target, lam_sched, clip=clip).run()
        for f in fields:
            assert np.all(f >= -clip - 1e-10)
            assert np.all(f <=  clip + 1e-10)

    def test_initial_field_is_smoothed(self):
        """Phi(x, 0) should have less variance than the raw target."""
        fields, _ = SRFLField(x, target, lam_sched).run()
        assert float(np.var(fields[0])) < float(np.var(target))

    def test_final_field_closer_to_target(self):
        """Phi(x, S) must be closer to f than Phi(x, 0)."""
        fields, _ = SRFLField(x, target, lam_sched).run()
        err0 = np.sqrt(np.mean((fields[0]  - target) ** 2))
        errS = np.sqrt(np.mean((fields[-1] - target) ** 2))
        assert errS < err0

    def test_final_field_method(self):
        f_final = SRFLField(x, target, lam_sched).final_field()
        assert f_final.shape == (len(x),)

    def test_convergence_rate_finite(self):
        engine  = SRFLField(x, target, lam_sched)
        _, errs = engine.run()
        r       = engine.convergence_rate(errs)
        assert np.isfinite(r)

    def test_repr(self):
        assert "SRFLField" in repr(SRFLField(x, target, lam_sched))
