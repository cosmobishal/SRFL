from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SRFLResult:
    x: np.ndarray
    scales: np.ndarray
    fields: list[np.ndarray]
    errors: np.ndarray
    actions: list[dict[str, float]]
    agent_counts: np.ndarray
    events: list[dict]
    candidates_by_scale: list[list]

    @property
    def final_field(self) -> np.ndarray:
        return self.fields[-1].copy()

    @property
    def initial_error(self) -> float:
        return float(self.errors[0])

    @property
    def final_error(self) -> float:
        return float(self.errors[-1])

    @property
    def improvement(self) -> float:
        return float(1.0 - self.final_error / max(self.initial_error, 1e-15))

    def predict(self, x_new: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(x_new, dtype=float), self.x, self.final_field)

    def save_npz(self, path: str) -> None:
        output = str(path)
        np.savez_compressed(output, x=self.x, scales=self.scales,
                            fields=np.asarray(self.fields), errors=self.errors,
                            agent_counts=self.agent_counts)
