# Design decisions

## Finite domain boundaries

A bounded signal should not silently become periodic.
The default reflective Gaussian operator therefore avoids wrap around at the endpoints.
Periodic filtering remains available for genuinely periodic signals.

## Adaptive structure

The defect detector uses robust statistics instead of a fixed second derivative threshold.
The median absolute deviation is less sensitive to isolated extreme values than a raw global standard deviation.
The curvature term is multiplied by `lambda^2`, which follows the standard scale normalization for second derivatives.

## Sparse correction

A defect is not injected at an arbitrary amplitude.
Its coefficient is the projection of the current residual onto the defect atom.
This gives every correction a direct numerical interpretation and makes the atom registry reusable.

## Swarm role

The swarm is deliberately separated from the residual projection.
This avoids hiding a second optimizer inside the method.
The swarm records structural support and supplies an interpretable event stream.

## Determinism

The release configuration has no random sampling in the core update.
A seed field remains in the configuration so future stochastic proposal strategies can be added without changing the public configuration shape.

## Scope

The current implementation is one dimensional.
This is intentional.
A two or three dimensional extension should define geometry aware defect atoms, memory behavior, and boundary conditions before being added to the same API.
