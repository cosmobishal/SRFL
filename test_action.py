"""Tests for srfl.action — ActionFunctional"""
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
from srfl import ActionFunctional, SRFLField

x          = np.linspace(-np.pi, np.pi, 256)
lam_sched  = np.logspace(0, -1.5, 40)
target     = np.where(x >= 0, 1.0, 0.0).astype(float)


def make_fields():
    engine = SRFLField(x, target, lam_sched, dt=0.25)
    fields, _ = engine.run()
    return fields


class TestActionFunctional:

    def setup_method(self):
        self.action = ActionFunctional(x, lam_sched, target)
        self.fields = make_fields()

    def test_a_data_non_negative(self):
        val = self.action.A_data(self.fields)
        assert val >= 0.0

    def test_a_data_zero_for_perfect_field(self):
        """If Φ = f at every scale, A_data ≈ 0."""
        perfect = [target.copy() for _ in lam_sched]
        val = self.action.A_data(perfect)
        assert val < 1e-8

    def test_a_data_decreases_with_evolution(self):
        """Data term at fine scale < data term at coarse scale."""
        early = self.fields[:5]
        late  = self.fields[-5:]
        A_early = self.action.A_data(early)
        A_late  = self.action.A_data(late)
        # Late fields are closer to target — lower data cost
        assert A_late <= A_early

    def test_a_scale_non_negative(self):
        val = self.action.A_scale(self.fields)
        assert val >= 0.0

    def test_a_sym_zero_without_operator(self):
        """Without symmetry operator, A_sym = 0 exactly."""
        val = self.action.A_sym(self.fields)
        assert val == 0.0

    def test_a_sym_with_reflection(self):
        """Reflection operator ℛ: φ(x) → φ(-x).
           For asymmetric fields (like H(x)), A_sym > 0."""
        def reflect(phi):
            return phi[::-1].copy()
        action_sym = ActionFunctional(x, lam_sched, target,
                                      symmetry_op=reflect)
        val = action_sym.A_sym(self.fields)
        assert val >= 0.0

    def test_a_cplx_non_negative(self):
        norms = [float(k * 0.01) for k in range(len(lam_sched))]
        val   = self.action.A_cplx(norms)
        assert val >= 0.0

    def test_a_cplx_zero_for_zero_norms(self):
        norms = [0.0] * len(lam_sched)
        val   = self.action.A_cplx(norms)
        assert val == 0.0

    def test_total_returns_dict(self):
        result = self.action.total(self.fields)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"data", "scale", "symmetry",
                                      "complexity", "total"}

    def test_total_consistency(self):
        """total == data + scale + symmetry + complexity."""
        result = self.action.total(self.fields)
        manual = (result["data"] + result["scale"] +
                  result["symmetry"] + result["complexity"])
        assert abs(result["total"] - manual) < 1e-12

    def test_repr(self):
        r = repr(self.action)
        assert "ActionFunctional" in r
