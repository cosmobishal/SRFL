# Novelty and prior art note

This project builds a specific combination of ideas.
The combination is the research contribution being developed here.
It should not be described as an isolated invention of every underlying ingredient.

## Established neighboring areas

Renormalization group ideas have already been connected to machine learning.
Koch-Janusz and Ringel showed an information theoretic neural approach that performs iterative renormalization group steps and identifies relevant degrees of freedom.

Non local operators are also established in machine learning.
Wang, Girshick, Gupta, and He introduced non local neural network blocks that aggregate information from all positions.

Renormalization group methods have also been used directly to study collective swarms.
Cavagna and collaborators developed a dynamical renormalization group treatment of collective swarm behavior.
Later work reported a nontrivial fixed point for natural swarms.

## Specific SRFL combination

The present repository combines the following pieces into one numerical learning procedure.

```text
scale indexed field state
        +
non local Gaussian residual transport
        +
scale normalized singularity detection
        +
sparse defect atom projection
        +
explicit defect registry
        +
swarm based structural support tracking
```

The adaptive defect projection stage is the main implementation step that turns the original conceptual prototype into a reusable numerical algorithm.

## Priority statement

A targeted search on 7 September 2026 did not locate a paper or public package using the exact name SRFL together with the same formulation.
This is not evidence that no related unpublished, non indexed, or independently developed system exists.
It is also not enough to establish a legal or publication priority claim.

The repository therefore uses the following wording.

> SRFL is presented as a proposed research framework whose exact combination was not found in the targeted literature and public code search performed for this release.

A future publication should include a broader systematic review before making a stronger first in world claim.

## References

See `references.bib` for persistent identifiers and bibliographic details.
