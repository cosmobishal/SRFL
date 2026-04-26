"""
srfl.swarm
==========
SRFL swarm model.

Agent state:  aᵢ = (xᵢ, λᵢ, Φᵢ, Dᵢ)

Update rules:
  - Motion     : xᵢ ← xᵢ + η·λ·(argmax_{|x-xᵢ|<δ} |∂²Φ| − xᵢ)
  - Spawn      : |∂²Φ(xᵢ)| > κ_spawn  AND  min_j |xⱼ-xᵢ| > δ_spawn
  - Merge      : |xᵢ-xⱼ| < ε_merge
  - Annihilate : |∂²Φ(xᵢ)| < κ_ann  OR  ‖Dᵢ‖_𝒟 < ε_ann
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Agent:
    """
    Single SRFL agent.

    Attributes
    ----------
    x      : float   — spatial position xᵢ ∈ Ω
    lam    : float   — local scale parameter λᵢ
    mass   : float   — L² mass ‖Φᵢ‖  (used in merge weighting)
    defect_norm : float — 𝒟-norm of carried defect
    alive  : bool    — True if not annihilated
    """
    x            : float
    lam          : float
    mass         : float  = 1.0
    defect_norm  : float  = 0.0
    alive        : bool   = True
    id           : int    = field(default_factory=lambda: Agent._counter())

    _count: int = 0

    @staticmethod
    def _counter() -> int:
        Agent._count += 1
        return Agent._count


# ─────────────────────────────────────────────────────────────────────────────
class Swarm:
    """
    SRFL agent swarm.

    Parameters
    ----------
    x               : np.ndarray  — spatial grid
    n_init          : int         — initial number of agents
    spawn_period    : int         — spawn check every n scale steps
    merge_eps       : float       — merge threshold ε_merge
    annihil_thresh  : float       — annihilation curvature threshold κ_ann
    annihil_norm    : float       — annihilation defect-norm threshold ε_ann
    mobility        : float       — agent motion coefficient η
    spawn_kappa     : float       — spawn curvature threshold κ_spawn
    delta_search    : float       — motion search radius δ
    """

    def __init__(
        self,
        x:              np.ndarray,
        n_init:         int   = 14,
        spawn_period:   int   = 8,
        merge_eps:      float = 0.10,
        annihil_thresh: float = 0.06,
        annihil_norm:   float = 0.01,
        mobility:       float = 0.25,
        spawn_kappa:    float = 1.4,
        delta_search:   float = 0.8,
    ):
        self.x               = x
        self.dx              = float(x[1] - x[0])
        self.N               = len(x)
        self.spawn_period    = spawn_period
        self.merge_eps       = merge_eps
        self.annihil_thresh  = annihil_thresh
        self.annihil_norm    = annihil_norm
        self.mobility        = mobility
        self.spawn_kappa     = spawn_kappa
        self.delta_search    = delta_search

        # Initialise agents uniformly
        positions = np.linspace(x[0] * 0.85, x[-1] * 0.85, n_init)
        self.agents: List[Agent] = [Agent(x=float(p), lam=1.0) for p in positions]

        # History tracking
        self.history: List[List[float]] = []
        self.events:  List[Tuple[int, str, float]] = []
        self._step_count = 0

    # ------------------------------------------------------------------ #
    # Curvature utilities                                                   #
    # ------------------------------------------------------------------ #

    def _curvature(self, phi: np.ndarray) -> np.ndarray:
        """Return |∂²Φ/∂x²| on the grid."""
        return np.abs(np.gradient(np.gradient(phi, self.dx), self.dx))

    def _curvature_at(self, pos: float, curv: np.ndarray) -> float:
        """Interpolate curvature at position pos."""
        return float(np.interp(pos, self.x, curv))

    def _best_in_ball(self, pos: float, curv: np.ndarray) -> float:
        """Return x-coordinate of curvature argmax in ball of radius δ."""
        mask = np.abs(self.x - pos) <= self.delta_search
        if not mask.any():
            return pos
        sub_x = self.x[mask]
        sub_c = curv[mask]
        return float(sub_x[np.argmax(sub_c)])

    # ------------------------------------------------------------------ #
    # Lifecycle rules                                                       #
    # ------------------------------------------------------------------ #

    def _move(self, curv: np.ndarray, lam: float) -> None:
        """Move agents toward local curvature maxima."""
        for a in self.agents:
            if not a.alive:
                continue
            best  = self._best_in_ball(a.x, curv)
            a.x   = float(np.clip(
                a.x + self.mobility * lam * (best - a.x),
                self.x[0], self.x[-1]))
            a.lam = lam

    def _spawn(self, curv: np.ndarray, phi: np.ndarray,
               target: np.ndarray, lam: float) -> None:
        """
        Spawn new agents at top-2 curvature peaks,
        provided no existing agent is already nearby.
        """
        peaks = np.argsort(curv)[-2:]
        alive_pos = np.array([a.x for a in self.agents if a.alive])

        for pidx in peaks:
            cx   = float(self.x[pidx])
            cv   = float(curv[pidx])
            if cv < self.spawn_kappa:
                continue
            if len(alive_pos) > 0 and np.min(np.abs(alive_pos - cx)) < self.merge_eps * 1.5:
                continue
            residual_norm = float(np.sqrt(np.mean((target - phi)**2)))
            child = Agent(x=cx, lam=lam * 0.6,
                          mass=residual_norm, defect_norm=cv * 0.05)
            self.agents.append(child)
            self.events.append((self._step_count, "spawn", cx))
            alive_pos = np.append(alive_pos, cx)

    def _merge(self) -> None:
        """Merge spatially coincident agents (centroid with mass weighting)."""
        merged  = set()
        new_agents: List[Agent] = []

        alive = [a for a in self.agents if a.alive]
        for i, ai in enumerate(alive):
            if i in merged:
                continue
            close = [j for j in range(i + 1, len(alive))
                     if j not in merged
                     and abs(alive[j].x - ai.x) < self.merge_eps]
            if close:
                group   = [ai] + [alive[j] for j in close]
                total_m = sum(ag.mass for ag in group)
                cx      = sum(ag.x * ag.mass for ag in group) / total_m
                cn      = sum(ag.defect_norm for ag in group)
                merged_agent = Agent(x=cx, lam=ai.lam,
                                     mass=total_m, defect_norm=cn)
                new_agents.append(merged_agent)
                self.events.append((self._step_count, "merge", cx))
                merged.update(close)
            else:
                new_agents.append(ai)

        self.agents = new_agents

    def _annihilate(self, curv: np.ndarray) -> None:
        """Remove agents in flat regions or with negligible defect content."""
        for a in self.agents:
            if not a.alive:
                continue
            local_curv = self._curvature_at(a.x, curv)
            if local_curv < self.annihil_thresh or \
               a.defect_norm < self.annihil_norm:
                a.alive = False
                self.events.append((self._step_count, "annihilate", a.x))

        # Ensure at least one agent survives
        alive = [a for a in self.agents if a.alive]
        if not alive:
            # Resurrect at global curvature maximum
            peak_x = float(self.x[np.argmax(curv)])
            self.agents = [Agent(x=peak_x, lam=self.agents[-1].lam)]
        else:
            self.agents = alive

    # ------------------------------------------------------------------ #
    # Main update                                                           #
    # ------------------------------------------------------------------ #

    def step(self, phi: np.ndarray, target: np.ndarray,
             lam: float, step_idx: int) -> None:
        """
        Perform one swarm update at pseudo-time step `step_idx`.

        Parameters
        ----------
        phi       : current field Φ(·, s)
        target    : target function f
        lam       : current scale λ
        step_idx  : integer step counter
        """
        self._step_count = step_idx
        curv = self._curvature(phi)

        self._move(curv, lam)

        if step_idx % self.spawn_period == 0:
            self._spawn(curv, phi, target, lam)

        self._merge()
        self._annihilate(curv)

        # Record history
        self.history.append([a.x for a in self.agents])

    def positions(self) -> List[float]:
        """Return current agent positions."""
        return [a.x for a in self.agents]

    def count(self) -> int:
        """Return current number of alive agents."""
        return len(self.agents)

    def interaction_matrix(self, lam: float,
                           sigma_lam: float = 0.1) -> np.ndarray:
        """
        Return the M×M inter-agent interaction matrix Kᵢⱼ.

        Kᵢⱼ = exp(-(xᵢ-xⱼ)²/(2λ²)) · exp(-(λᵢ-λⱼ)²/(2σ_λ²))
        """
        pos  = np.array([a.x   for a in self.agents])
        lams = np.array([a.lam for a in self.agents])
        d_x  = (pos[:, None]  - pos[None, :])**2 / (2 * lam**2)
        d_l  = (lams[:, None] - lams[None, :])**2 / (2 * sigma_lam**2)
        return np.exp(-d_x - d_l)

    def event_summary(self) -> dict:
        """Return counts of each lifecycle event type."""
        summary = {"spawn": 0, "merge": 0, "annihilate": 0}
        for _, etype, _ in self.events:
            summary[etype] += 1
        return summary

    def __repr__(self) -> str:
        return (f"Swarm(agents={self.count()}, "
                f"events={len(self.events)}, "
                f"steps={self._step_count})")
