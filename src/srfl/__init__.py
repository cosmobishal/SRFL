"""Swarm Renormalization Field Learning.

A non gradient field learning method built from scale indexed non local
transport, sparse defect projection, and swarm based support tracking.
"""

from .action import ActionFunctional
from .defects import (
    DefectAlgebra,
    DefectAtom,
    DefectCandidate,
    DefectDetector,
    DefectRegistry,
    GaussianBumpAtom,
    OscillatoryAtom,
    PiecewiseAtom,
    StepAtom,
)
from .grid import Grid1D
from .kernel import GaussianKernel
from .result import SRFLResult
from .solver import SRFLConfig, SRFLRegressor, SRFLSolver
from .swarm import Agent, Swarm

__version__ = "2.0.0"
__all__ = [
    "ActionFunctional", "Agent", "DefectAlgebra", "DefectAtom", "DefectCandidate",
    "DefectDetector", "DefectRegistry", "GaussianBumpAtom", "GaussianKernel", "Grid1D",
    "OscillatoryAtom", "PiecewiseAtom", "SRFLConfig", "SRFLRegressor", "SRFLResult",
    "SRFLSolver", "StepAtom", "Swarm",
]
