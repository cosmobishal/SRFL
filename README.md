# SRFL

Swarm Renormalization Field Learning is a non gradient learning method for one dimensional signals.
It treats the learned object as a field that moves through a sequence of spatial scales.
The smooth part is updated by a non local Gaussian operator and unresolved structure is represented with sparse defect atoms.
A small swarm tracks where those structural corrections are needed.

This repository develops that idea into a reusable algorithm called adaptive defect projection.
The solver has four coupled stages.

1. The target is initialized at a coarse scale with a finite domain Gaussian operator.
2. A non local residual transport step advances the smooth background.
3. A scale normalized detector identifies candidate structural regions and classifies them as step, oscillatory, piecewise, or smooth bump defects.
4. Each candidate is projected onto the current residual and the swarm updates its sparse support map.

The projection stage is deliberately coefficient based rather than gradient based.
For a defect atom `d`, its coefficient is computed from the residual inner product

```text
alpha = <r, d> / <d, d>
```

This makes the method usable as a deterministic numerical algorithm rather than as a training loop around adjustable model parameters.


## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e .[test,plot]
```

## Run a complete example

```bash
srfl --target step --n 512 --scales 64 --out results/step
```

You can also run the Python examples directly.

```bash
python examples/step.py
python examples/oscillatory.py
```

## Python API

```python
import numpy as np
from srfl import SRFLRegressor

x = np.linspace(-np.pi, np.pi, 512)
y = (x >= 0).astype(float)

model = SRFLRegressor(
    scales=64,
    lam_max=1.0,
    lam_min=0.02,
    dt=0.4,
    n_agents=8,
)
model.fit(x, y)

prediction = model.predict(x)
print(model.result_.final_error)
```

The result object exposes the full scale history.

```python
result = model.result_
print(result.scales)
print(result.errors)
print(result.agent_counts)
print(result.events)
```

## Algorithm at a glance

Let `phi_k` be the field at scale `lambda_k` and let `r_k = y - phi_k`.
The smooth update is

```text
u_k = K_lambda_k * r_k
phi_(k+1/2) = phi_k + dt * u_k
```

A detector then builds a scale normalized geometry score from curvature and residual structure.
For every selected candidate `c`, the registry returns an atom `d_c` and the solver computes

```text
alpha_c = <y - phi_(k+1/2), d_c> / <d_c, d_c>
phi_(k+1) = phi_(k+1/2) + gain * alpha_c * d_c
```

The swarm updates positions toward the selected candidate supports.
Agents spawn when a candidate exceeds the detector threshold and merge when their supports become spatially coincident.
Weak agents decay naturally.

This design gives the method a clear separation between field transport, structural representation, and support management.

## Complexity

The default finite domain Gaussian operator uses SciPy's one dimensional Gaussian filter.
For a grid of `N` points the filtering cost is approximately linear in `N` for a fixed kernel truncation.
Candidate detection is linear in the number of grid points for the standard case.
The number of defect projections is bounded by `max_defects` per scale.

## Tests

The repository includes tests for boundary behavior, Gaussian scale composition, detector output, defect rendering, swarm lifecycle, solver determinism, regression style prediction, and end to end error reduction.

```bash
pytest
```

The release process also builds the wheel and runs a command line smoke test.



## References

The design is informed by established work on renormalization group methods, machine learned renormalization, non local operations, and field theoretic descriptions of swarms.
See `docs/references.bib` and `docs/novelty.md`.

## License

MIT
