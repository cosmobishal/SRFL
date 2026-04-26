"""Tests for srfl.field — SRFLField and SingularityGenerator"""
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
from srfl import SRFLField
from srfl.field import SingularityGenerator

x         = np.linspace(-np.pi, np.pi, 256)
dx        = float(x[1] - x[0])
lam_sched = np.logspace(0, -1.5, 30)
target    = np.where(x >= 0, 1.0, 0.0).astype(float)


class TestSingularityGenerator:

    def test_output_shape(self):
        S   = SingularityGenerator(kappa=1.8, eps=0.07)
        phi = np.sin(x)
        out = S(phi, dx)
        assert out.shape == (len(x),)

    def test_zero_for_smooth_field(self):
        """Slowly varying field → curvature < κ → 𝒮 = 0."""
        S   = SingularityGenerator(kappa=10.0, eps=0.07)
        phi = 0.01 * np.sin(x)          # tiny curvature
        out = S(phi, dx)
        assert np.allclose(out, 0.0)

    def test_nonzero_for_steep_field(self):
        """Steep step → large curvature → 𝒮 fires."""
        S   = SingularityGenerator(kappa=0.5, eps=0.1)
        phi = np.where(x >= 0, 1.0, 0.0).astype(float)
        out = S(phi, dx)
        assert np.any(out != 0.0)

    def test_bounded_output(self):
        """𝒮 uses tanh internally → output bounded."""
        S   = SingularityGenerator(kappa=0.1, eps=0.5)
        phi = np.random.randn(len(x)) * 10   # wild field
        out = S(phi, dx)
        assert np.all(np.isfinite(out))

    def test_sign_alignment(self):
        """𝒮 should be aligned with sign of ∂²Φ."""
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
        bad_sched = np.linspace(0.01, 1.0, 20)   # increasing!
        with pytest.raises(ValueError):
            SRFLField(x, target, bad_sched)

    def test_run_returns_correct_lengths(self):
        engine       = SRFLField(x, target, lam_sched)
        fields, errs = engine.run()
        assert len(fields) == len(lam_sched)
        assert len(errs)   == len(lam_sched)

    def test_fields_have_correct_shape(self):
        engine  = SRFLField(x, target, lam_sched)
        fields, _ = engine.run()
        for f in fields:
            assert f.shape == (len(x),)

    def test_errors_non_negative(self):
        engine = SRFLField(x, target, lam_sched)
        _, errs = engine.run()
        assert all(e >= 0.0 for e in errs)

    def test_error_converges(self):
        """Final L² error must be strictly less than initial."""
        engine       = SRFLField(x, target, lam_sched)
        _, errs      = engine.run()
        assert errs[-1] < errs[0]

    def test_field_clipped(self):
        """All field values must lie within [-clip, clip]."""
        clip   = 2.8
        engine = SRFLField(x, target, lam_sched, clip=clip)
        fields, _ = engine.run()
        for f in fields:
            assert np.all(f >= -clip - 1e-10)
            assert np.all(f <=  clip + 1e-10)

    def test_initial_field_is_smoothed(self):
        """Initial field Φ(·,0) should be smoother than raw target."""
        engine = SRFLField(x, target, lam_sched)
        fields, _ = engine.run()
        init_var   = float(np.var(fields[0]))
        target_var = float(np.var(target))
        assert init_var < target_var

    def test_final_field_closer_to_target(self):
        """Φ(·, S) should be closer to f than Φ(·, 0)."""
        engine = SRFLField(x, target, lam_sched)
        fields, _ = engine.run()
        err0 = np.sqrt(np.mean((fields[0]  - target)**2))
        errS = np.sqrt(np.mean((fields[-1] - target)**2))
        assert errS < err0

    def test_final_field_method(self):
        engine = SRFLField(x, target, lam_sched)
        f_final = engine.final_field()
        assert f_final.shape == (len(x),)

    def test_convergence_rate_finite(self):
        engine = SRFLField(x, target, lam_sched)
        _, errs = engine.run()
        r = engine.convergence_rate(errs)
        assert np.isfinite(r)

    def test_repr(self):
        engine = SRFLField(x, target, lam_sched)
        assert "SRFLField" in repr(engine)
