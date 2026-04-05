"""
Swarm Renormalization Field Learning (SRFL)
===========================================
A non-local, multi-scale, defect-driven learning paradigm.

Author : Bishal Neupane <cosmobishal@gmail.com>
Affil  : Astronomy Squad of Koshi
Year   : 2026
License: MIT

Modules
-------
kernel      : Gaussian non-local kernel K(x, x', λ)
field       : SRFL field evolution engine
defects     : Defect algebra  {Ŝ, Ô, Ĉ}
swarm       : Agent swarm model (spawn / merge / annihilate)
action      : Action functional 𝒜 = 𝒜_data + 𝒜_scale + 𝒜_sym + 𝒜_cplx
multiscale  : Scale projection Π(λ₁ → λ₂) and consistency checks
"""

from .kernel import SRFLKernel
from .field import SRFLField
from .defects import StepDefect, OscillatoryDefect, ConditionalDefect, DefectAlgebra
from .swarm import Agent, Swarm
from .action import ActionFunctional
from .multiscale import ScaleProjection

__version__ = "1.0.0"
__author__  = "Bishal Neupane"
__email__   = "cosmobishal@gmail.com"

__all__ = [
    "SRFLKernel",
    "SRFLField",
    "StepDefect",
    "OscillatoryDefect",
    "ConditionalDefect",
    "DefectAlgebra",
    "Agent",
    "Swarm",
    "ActionFunctional",
    "ScaleProjection",
]
