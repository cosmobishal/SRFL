# Algorithm specification

## 1. State

The solver maintains a field `phi(x, lambda)` on a validated uniform grid.
The scale schedule is strictly decreasing from a coarse scale to a fine scale.
The target is fixed throughout the run.

## 2. Non local transport

At scale `lambda`, the residual is

```text
r = y - phi
```

The background update is

```text
u = K_lambda(r)
phi <- phi + dt * u
```

The default kernel is Gaussian with reflective finite domain boundaries.
A periodic `wrap` mode is available when the underlying problem is periodic.

## 3. Geometry score

The detector computes

```text
d2 = d^2 phi / dx^2
q_curv = |lambda^2 d2| / robust_scale(lambda^2 d2)
q_res  = |r| / robust_scale(r)
q      = max(q_curv, q_res)
```

The scale factor makes curvature comparable across the scale schedule.
The robust scale uses the median absolute deviation.

Oscillation evidence is measured with a local zero crossing density of the residual.
A high crossing density together with non negligible residual amplitude promotes the oscillatory atom.
Sharp high curvature promotes the step atom.
Other selected structures use the piecewise atom.
The Gaussian bump atom is part of the registry for custom extensions even when the detector does not select it automatically.

## 4. Defect projection

For a candidate `c`, the registry creates a sampled atom `d_c`.
The coefficient is obtained by a least squares projection of the residual onto that single atom.
No parameter gradient is used.
The coefficient is clipped only to protect the finite difference state from pathological residuals.

## 5. Swarm support map

Agents store position, active scale, structural strength, defect type, and age.
Agents move toward the nearest selected structural candidate.
A strong candidate can spawn a new agent when there is no nearby support.
Nearby agents merge into a strength weighted centroid.
Agent strength decays between updates and old weak agents disappear.

The swarm is therefore a support bookkeeping mechanism rather than a second optimizer hidden inside the solver.

## 6. Output

`SRFLResult` records

- all field snapshots
- scale values
- root mean square errors
- finite scale action diagnostics
- agent population history
- lifecycle events
- fitted defect candidates

The result can be exported to a compressed NumPy archive.

## 7. Why this formulation is useful

The algorithm separates smooth transport from structural correction.
This lets the same field engine work with a small defect dictionary, a user supplied dictionary, or a problem specific registry.
The swarm layer can be inspected independently of the field state.
That makes ablation studies straightforward.

Recommended ablations compare the full solver against

```text
A. Gaussian transport only
B. Gaussian transport plus fixed threshold defect injection
C. Gaussian transport plus adaptive defect projection
D. Full adaptive defect projection plus swarm support tracking
```

The repository currently implements the full method and exposes the components needed to implement the ablations without rewriting the solver core.
