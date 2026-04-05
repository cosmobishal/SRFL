"""Tests for srfl.swarm"""
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
from srfl import Swarm

x      = np.linspace(-np.pi, np.pi, 256)
target = np.where(x >= 0, 1.0, 0.0).astype(float)
phi0   = np.zeros_like(x)


class TestSwarm:

    def setup_method(self):
        self.swarm = Swarm(x, n_init=8, spawn_period=5,
                           merge_eps=0.15, annihil_thresh=0.05)

    def test_initial_count(self):
        assert self.swarm.count() == 8

    def test_step_runs_without_error(self):
        self.swarm.step(phi0, target, lam=0.5, step_idx=1)

    def test_count_non_negative_after_steps(self):
        for k in range(20):
            self.swarm.step(phi0, target, lam=0.5*(1-k/40), step_idx=k)
        assert self.swarm.count() >= 1

    def test_positions_in_domain(self):
        for k in range(10):
            self.swarm.step(phi0, target, lam=0.4, step_idx=k)
        for p in self.swarm.positions():
            assert x[0] <= p <= x[-1]

    def test_history_length(self):
        n_steps = 15
        for k in range(n_steps):
            self.swarm.step(phi0, target, lam=0.5, step_idx=k)
        assert len(self.swarm.history) == n_steps

    def test_event_types(self):
        for k in range(30):
            self.swarm.step(phi0, target, lam=0.4*(1-k/60), step_idx=k)
        for (s, etype, ex) in self.swarm.events:
            assert etype in {"spawn", "merge", "annihilate"}
            assert x[0] <= ex <= x[-1]

    def test_event_summary_keys(self):
        summary = self.swarm.event_summary()
        assert set(summary.keys()) == {"spawn", "merge", "annihilate"}

    def test_interaction_matrix_shape(self):
        M  = self.swarm.interaction_matrix(lam=0.5)
        n  = self.swarm.count()
        assert M.shape == (n, n)

    def test_interaction_matrix_diagonal_ones(self):
        M = self.swarm.interaction_matrix(lam=0.5)
        assert np.allclose(np.diag(M), 1.0)

    def test_interaction_matrix_symmetric(self):
        M = self.swarm.interaction_matrix(lam=0.5)
        assert np.allclose(M, M.T, atol=1e-10)

    def test_repr(self):
        r = repr(self.swarm)
        assert "Swarm" in r
