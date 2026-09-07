from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .action import ActionFunctional
from .defects import DefectCandidate, DefectDetector, DefectRegistry
from .grid import Grid1D
from .kernel import GaussianKernel
from .result import SRFLResult
from .swarm import Swarm


@dataclass
class SRFLConfig:
    scales: int = 64
    lam_max: float = 1.0
    lam_min: float = 0.02
    dt: float = 0.4
    defect_gain: float = 0.55
    detector_sensitivity: float = 4.0
    max_defects: int = 6
    n_agents: int = 8
    boundary: str = "reflect"
    clip_margin: float = 0.15
    seed: int = 0

    def validate(self) -> None:
        if self.scales < 2:
            raise ValueError("scales must be at least 2")
        if self.lam_max <= self.lam_min or self.lam_min <= 0:
            raise ValueError("lam_max must exceed positive lam_min")
        if not (0 < self.dt <= 1.0):
            raise ValueError("dt must be in (0, 1]")
        if not (0 < self.defect_gain <= 2):
            raise ValueError("defect_gain must be in (0, 2]")
        if self.n_agents < 0:
            raise ValueError("n_agents cannot be negative")
        if self.clip_margin < 0:
            raise ValueError("clip_margin cannot be negative")
        if self.boundary not in {"reflect", "wrap"}:
            raise ValueError("unsupported boundary mode")


class SRFLSolver:
    """Adaptive Swarm Renormalization Field Learning solver.

    The update has two pieces. A non local scale controlled residual transport
    step updates the smooth background. A sparse defect projection then places
    only the structural atoms required by the current residual geometry.
    """

    def __init__(self, config: Optional[SRFLConfig] = None,
                 registry: Optional[DefectRegistry] = None) -> None:
        self.config = config or SRFLConfig()
        self.config.validate()
        self.registry = registry or DefectRegistry()

    @staticmethod
    def scales(lam_max: float, lam_min: float, n: int) -> np.ndarray:
        return np.geomspace(lam_max, lam_min, n)

    def _project_atom(self, x: np.ndarray, residual: np.ndarray,
                      candidate: DefectCandidate) -> tuple[DefectCandidate, np.ndarray]:
        atom = self.registry.evaluate(x, candidate)
        denom = float(np.dot(atom, atom))
        if denom <= 1e-18:
            return candidate.with_amplitude(0.0), atom
        amplitude = float(np.dot(residual, atom) / denom)
        amplitude = float(np.clip(amplitude, -2.0, 2.0))
        return candidate.with_amplitude(amplitude), atom

    def fit(self, x: np.ndarray, target: np.ndarray) -> SRFLResult:
        rng = np.random.default_rng(self.config.seed)
        del rng  # the solver is deterministic; the seed is retained for API stability
        grid = Grid1D(x)
        y = grid.check_values(target, "target")
        scales = self.scales(self.config.lam_max, self.config.lam_min, self.config.scales)
        detector = DefectDetector(self.config.detector_sensitivity,
                                  self.config.max_defects,
                                  min_separation=max(grid.dx, 0.25 * self.config.lam_min))
        swarm = Swarm(grid.x, n_init=self.config.n_agents)

        initial_kernel = GaussianKernel(grid, scales[0], self.config.boundary)
        phi = initial_kernel.convolve(y)
        fields = [phi.copy()]
        errors = [float(np.sqrt(np.mean((phi - y) ** 2)))]
        candidates_by_scale: list[list[DefectCandidate]] = [[]]
        actions: list[dict[str, float]] = []
        agent_counts = [swarm.count()]

        action_fn = ActionFunctional(grid.x, scales, y)

        for step, lam in enumerate(scales[1:], start=1):
            kernel = GaussianKernel(grid, float(lam), self.config.boundary)
            residual = y - phi
            background = kernel.convolve(residual)
            phi = phi + self.config.dt * background

            candidates = detector.detect(grid.x, phi, y - phi, float(lam))
            applied: list[DefectCandidate] = []
            for candidate in candidates:
                fitted, atom = self._project_atom(grid.x, y - phi, candidate)
                correction = self.config.defect_gain * fitted.amplitude * atom
                phi = phi + correction
                applied.append(fitted)

            lo = float(np.min(y) - self.config.clip_margin)
            hi = float(np.max(y) + self.config.clip_margin)
            phi = np.clip(phi, lo, hi)

            swarm.update(applied, float(lam), step)
            fields.append(phi.copy())
            err = float(np.sqrt(np.mean((phi - y) ** 2)))
            errors.append(err)
            candidates_by_scale.append(applied)
            agent_counts.append(swarm.count())

            active_norm = sum(abs(c.amplitude) for c in applied)
            window = min(len(fields), 8)
            window_action = ActionFunctional(grid.x, scales[-window:], y)
            actions.append(window_action.total(fields[-window:],
                                                lambda scale: GaussianKernel(grid, scale, self.config.boundary),
                                                [active_norm] * window))

        return SRFLResult(grid.x.copy(), scales, fields, np.asarray(errors), actions,
                          np.asarray(agent_counts), swarm.events, candidates_by_scale)


class SRFLRegressor:
    """Scikit compatible style facade around :class:`SRFLSolver`."""

    def __init__(self, **kwargs) -> None:
        self.config = SRFLConfig(**kwargs) if kwargs else SRFLConfig()
        self.solver = SRFLSolver(self.config)
        self.result_: Optional[SRFLResult] = None
        self.x_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SRFLRegressor":
        self.result_ = self.solver.fit(x, y)
        self.x_ = self.result_.x.copy()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.result_ is None:
            raise RuntimeError("fit must be called before predict")
        return self.result_.predict(x)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(x)
        y = np.asarray(y, dtype=float)
        denom = float(np.sum((y - np.mean(y)) ** 2))
        return float(1.0 - np.sum((y - pred) ** 2) / max(denom, 1e-15))

    def __repr__(self) -> str:
        return f"SRFLRegressor(scales={self.config.scales}, lam=[{self.config.lam_min}, {self.config.lam_max}])"
