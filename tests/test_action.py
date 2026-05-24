"""Tests for srfl.action -- ActionFunctional"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srfl import ActionFunctional, SRFLField

x         = np.linspace(-np.pi, np.pi, 256)
lam_sched = np.logspace(0, -1.5, 40)
target    = np.where(x >= 0, 1.0, 0.0).astype(float)


@pytest.fixture(scope="module")
def fields():
    engine = SRFLField(x, target, lam_sched, dt=0.25)
    fs, _  = engine.run()
    return fs


@pytest.fixture(scope="module")
def action():
    return ActionFunctional(x, lam_sched, target)


class TestActionFunctional:

    def test_a_data_non_negative(self, fields, action):
        assert action.A_data(fields) >= 0.0

    def test_a_data_zero_for_perfect_field(self, action):
        """If Phi = f at every scale, A_data is essentially 0."""
        perfect = [target.copy() for _ in lam_sched]
        assert action.A_data(perfect) < 1e-8

    def test_a_data_decreases_with_evolution(self, fields, action):
        """Data term at fine scale is lower than at coarse scale."""
        assert action.A_data(fields[-5:]) <= action.A_data(fields[:5])

    def test_a_scale_non_negative(self, fields, action):
        assert action.A_scale(fields) >= 0.0

    def test_a_sym_zero_without_operator(self, fields, action):
        assert action.A_sym(fields) == 0.0

    def test_a_sym_with_reflection(self, fields):
        """Reflection operator R: phi(x) -> phi(-x). A_sym > 0 for H(x)."""
        def reflect(phi):
            return phi[::-1].copy()
        a = ActionFunctional(x, lam_sched, target, symmetry_op=reflect)
        assert a.A_sym(fields) >= 0.0

    def test_a_cplx_non_negative(self, action):
        norms = [float(k * 0.01) for k in range(len(lam_sched))]
        assert action.A_cplx(norms) >= 0.0

    def test_a_cplx_zero_for_zero_norms(self, action):
        assert action.A_cplx([0.0] * len(lam_sched)) == 0.0

    def test_total_returns_dict(self, fields, action):
        result = action.total(fields)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"data", "scale", "symmetry", "complexity", "total"}

    def test_total_consistency(self, fields, action):
        """total == data + scale + symmetry + complexity."""
        r      = action.total(fields)
        manual = r["data"] + r["scale"] + r["symmetry"] + r["complexity"]
        assert abs(r["total"] - manual) < 1e-12

    def test_repr(self, action):
        assert "ActionFunctional" in repr(action)
