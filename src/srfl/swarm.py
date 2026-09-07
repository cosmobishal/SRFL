from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional
import numpy as np

from .defects import DefectCandidate


@dataclass
class Agent:
    id: int
    position: float
    scale: float
    strength: float = 0.0
    kind: str = "background"
    age: int = 0


class Swarm:
    """Sparse support manager for defect regions.

    The swarm does not optimize parameters. It tracks where the field needs
    structural corrections and controls spawn, merge, drift, and decay.
    """

    def __init__(self, x: np.ndarray, n_init: int = 8, spawn_threshold: float = 4.0,
                 merge_distance: Optional[float] = None, decay: float = 0.92,
                 mobility: float = 0.35) -> None:
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size < 8:
            raise ValueError("x must be a one dimensional grid")
        if n_init < 0 or spawn_threshold <= 0 or not (0 < decay <= 1) or not (0 <= mobility <= 1):
            raise ValueError("invalid swarm parameters")
        self.x = x.copy()
        self.dx = float(x[1] - x[0])
        self.spawn_threshold = float(spawn_threshold)
        self.merge_distance = float(merge_distance if merge_distance is not None else 3.0 * self.dx)
        self.decay = float(decay)
        self.mobility = float(mobility)
        positions = np.linspace(x[0], x[-1], n_init + 2)[1:-1] if n_init else []
        self.agents = [Agent(i, float(p), 1.0) for i, p in enumerate(positions)]
        self.events: list[dict] = []
        self.history: list[list[Agent]] = []
        self._next_id = len(self.agents)

    def _spawn(self, candidate: DefectCandidate, lam: float, step: int) -> None:
        nearby = [a for a in self.agents if abs(a.position - candidate.center) < self.merge_distance]
        if candidate.score < self.spawn_threshold or nearby:
            return
        agent = Agent(self._next_id, candidate.center, lam, abs(candidate.amplitude), candidate.kind)
        self._next_id += 1
        self.agents.append(agent)
        self.events.append({"step": step, "type": "spawn", "position": agent.position,
                            "kind": agent.kind})

    def _merge(self, step: int) -> None:
        self.agents.sort(key=lambda a: a.position)
        merged: list[Agent] = []
        for agent in self.agents:
            if not merged or abs(agent.position - merged[-1].position) >= self.merge_distance:
                merged.append(agent)
                continue
            anchor = merged[-1]
            weight_a = max(abs(anchor.strength), 1e-12)
            weight_b = max(abs(agent.strength), 1e-12)
            total = weight_a + weight_b
            anchor.position = (weight_a * anchor.position + weight_b * agent.position) / total
            anchor.strength += agent.strength
            anchor.scale = min(anchor.scale, agent.scale)
            anchor.age = min(anchor.age, agent.age)
            self.events.append({"step": step, "type": "merge", "position": anchor.position,
                                "kind": anchor.kind})
        self.agents = merged

    def update(self, candidates: Iterable[DefectCandidate], lam: float, step: int) -> None:
        candidates = list(candidates)
        for agent in self.agents:
            agent.age += 1
            agent.strength *= self.decay
            nearest = min(candidates, key=lambda c: abs(c.center - agent.position), default=None)
            if nearest is not None:
                agent.position += self.mobility * (nearest.center - agent.position)
                agent.position = float(np.clip(agent.position, self.x[0], self.x[-1]))
                agent.scale = lam
                agent.strength = max(agent.strength, abs(nearest.amplitude))
                agent.kind = nearest.kind
        for candidate in candidates:
            self._spawn(candidate, lam, step)
        self._merge(step)
        self.agents = [a for a in self.agents if a.strength > 1e-8 or a.age < 3]
        self.history.append([Agent(a.id, a.position, a.scale, a.strength, a.kind, a.age) for a in self.agents])

    def positions(self) -> np.ndarray:
        return np.asarray([a.position for a in self.agents], dtype=float)

    def count(self) -> int:
        return len(self.agents)

    def interaction_matrix(self, lam: float, sigma_lam: float = 0.1) -> np.ndarray:
        if lam <= 0 or sigma_lam <= 0:
            raise ValueError("lam and sigma_lam must be positive")
        if not self.agents:
            return np.zeros((0, 0), dtype=float)
        pos = np.asarray([a.position for a in self.agents])
        scales = np.asarray([a.scale for a in self.agents])
        return np.exp(-0.5 * ((pos[:, None] - pos[None, :]) / lam) ** 2
                      - 0.5 * ((scales[:, None] - scales[None, :]) / sigma_lam) ** 2)

    def event_summary(self) -> dict[str, int]:
        out = {"spawn": 0, "merge": 0}
        for event in self.events:
            out[event["type"]] = out.get(event["type"], 0) + 1
        return out

    def __repr__(self) -> str:
        return f"Swarm(agents={len(self.agents)}, events={len(self.events)})"
