"""Tests for srfl.swarm -- Agent, Swarm"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srfl import Swarm

x      = np.linspace(-np.pi, np.pi, 256)
target = np.where(x >= 0, 1.0, 0.0).astype(float)
phi0   = np.zeros_like(x)


@pytest.fixture
def swarm():
    return Swarm(x, n_init=8, spawn_period=5, merge_eps=0.15, annihil_thresh=0.05)


class TestSwarm:

    def test_initial_count(self, swarm):
        assert swarm.count() == 8

    def test_step_runs_without_error(self, swarm):
        swarm.step(phi0, target, lam=0.5, step_idx=1)

    def test_count_non_negative_after_steps(self, swarm):
        for k in range(20):
            swarm.step(phi0, target, lam=0.5 * (1 - k / 40), step_idx=k)
        assert swarm.count() >= 1

    def test_positions_in_domain(self, swarm):
        for k in range(10):
            swarm.step(phi0, target, lam=0.4, step_idx=k)
        for p in swarm.positions():
            assert x[0] <= p <= x[-1]

    def test_history_length(self, swarm):
        n_steps = 15
        for k in range(n_steps):
            swarm.step(phi0, target, lam=0.5, step_idx=k)
        assert len(swarm.history) == n_steps

    def test_event_types(self, swarm):
        for k in range(30):
            swarm.step(phi0, target, lam=0.4 * (1 - k / 60), step_idx=k)
        for _, etype, ex in swarm.events:
            assert etype in {"spawn", "merge", "annihilate"}
            assert x[0] <= ex <= x[-1]

    def test_event_summary_keys(self, swarm):
        assert set(swarm.event_summary().keys()) == {"spawn", "merge", "annihilate"}

    def test_interaction_matrix_shape(self, swarm):
        M = swarm.interaction_matrix(lam=0.5)
        n = swarm.count()
        assert M.shape == (n, n)

    def test_interaction_matrix_diagonal_ones(self, swarm):
        M = swarm.interaction_matrix(lam=0.5)
        assert np.allclose(np.diag(M), 1.0)

    def test_interaction_matrix_symmetric(self, swarm):
        M = swarm.interaction_matrix(lam=0.5)
        assert np.allclose(M, M.T, atol=1e-10)

    def test_repr(self, swarm):
        assert "Swarm" in repr(swarm)
