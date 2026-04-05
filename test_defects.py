"""Tests for srfl.defects"""
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
from srfl import StepDefect, OscillatoryDefect, ConditionalDefect, DefectAlgebra

x  = np.linspace(-2.0, 2.0, 512)
dx = float(x[1] - x[0])


class TestStepDefect:

    def test_field_shape(self):
        D = StepDefect(x0=0.0, alpha=1.0)
        assert D.field(x).shape == (len(x),)

    def test_heaviside_values(self):
        D   = StepDefect(x0=0.0, alpha=1.0)
        out = D.field(x)
        assert np.all(out[x < 0]  == 0.0)
        assert np.all(out[x >= 0] == 1.0)

    def test_apply_additive(self):
        D   = StepDefect(x0=0.0, alpha=2.0)
        phi = np.ones_like(x)
        out = D.apply(x, phi)
        assert np.allclose(out[x >= 0], 3.0)

    def test_norm(self):
        D = StepDefect(x0=0.0, alpha=-3.5)
        assert D.norm() == 3.5

    def test_compose_same_x0(self):
        D1 = StepDefect(x0=0.0, alpha=1.0)
        D2 = StepDefect(x0=0.0, alpha=2.0)
        D3 = D1.compose(D2)
        assert D3.alpha == 3.0

    def test_compose_different_x0_raises(self):
        D1 = StepDefect(x0=0.0, alpha=1.0)
        D2 = StepDefect(x0=1.0, alpha=1.0)
        with pytest.raises(ValueError):
            D1.compose(D2)


class TestOscillatoryDefect:

    def test_zero_at_origin(self):
        D   = OscillatoryDefect(eps=0.5, beta=1.0)
        out = D.field(x)
        idx = np.argmin(np.abs(x))   # closest to 0
        assert abs(out[idx]) < 1e-10

    def test_zero_outside_support(self):
        D   = OscillatoryDefect(eps=0.3, beta=1.0)
        out = D.field(x)
        assert np.all(out[np.abs(x) >= 0.3] == 0.0)

    def test_nonzero_inside_support(self):
        D   = OscillatoryDefect(eps=0.5, beta=1.0)
        out = D.field(x)
        assert np.any(out[np.abs(x) < 0.5] != 0.0)

    def test_shape(self):
        D = OscillatoryDefect(eps=0.5, beta=2.0)
        assert D.field(x).shape == (len(x),)

    def test_negative_eps_raises(self):
        with pytest.raises(ValueError):
            OscillatoryDefect(eps=-0.1, beta=1.0)

    def test_apply(self):
        D   = OscillatoryDefect(eps=0.5, beta=1.0)
        phi = np.zeros_like(x)
        out = D.apply(x, phi)
        assert np.allclose(out, D.field(x))


class TestConditionalDefect:

    def test_piecewise_constant(self):
        D   = ConditionalDefect([(-1.0, 0.0), (0.0, 1.0)], [2.0, -1.0])
        out = D.field(x)
        assert np.allclose(out[(x >= -1.0) & (x < 0.0)], 2.0)
        assert np.allclose(out[(x >= 0.0)  & (x < 1.0)], -1.0)
        assert np.all(out[x < -1.0] == 0.0)

    def test_norm(self):
        D = ConditionalDefect([(-1,0),(0,1)], [3.0, -4.0])
        assert D.norm() == 7.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            ConditionalDefect([(0,1)], [1.0, 2.0])


class TestDefectAlgebra:

    def setup_method(self):
        self.alg = DefectAlgebra(x)

    def test_compose_fields_shape(self):
        D1  = StepDefect(x0=0.0, alpha=1.0)
        D2  = StepDefect(x0=0.0, alpha=0.5)
        phi = np.zeros_like(x)
        out = self.alg.compose_fields(D1, D2, phi)
        assert out.shape == (len(x),)

    def test_commutator_zero_for_same_defect(self):
        D   = StepDefect(x0=0.0, alpha=1.0)
        phi = np.zeros_like(x)
        com = self.alg.commutator_field(D, D, phi)
        assert np.allclose(com, 0.0)

    def test_total_norm(self):
        d_list = [StepDefect(alpha=2.0), OscillatoryDefect(eps=0.4, beta=3.0)]
        norm   = self.alg.total_norm(d_list)
        assert abs(norm - (2.0 + 3.0 * 0.4)) < 1e-10

    def test_detect_from_curvature_returns_list(self):
        phi  = np.where(x >= 0, 1.0, 0.0).astype(float)
        defs = DefectAlgebra.detect_from_curvature(x, phi, dx)
        assert isinstance(defs, list)
