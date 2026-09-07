# Contributing

Contributions should keep the numerical core deterministic and testable.
New defect atoms should implement the small `DefectAtom` interface and include tests for finite output and expected support behavior.

Changes to the solver should include a regression test that exercises the public API.
Mathematical assumptions should be stated in the docstring of the affected class or function.

Before opening a pull request run

```bash
pytest
python -m pip wheel . --no-build-isolation --no-deps -w dist
python examples/step.py
python examples/oscillatory.py
```

Avoid adding heavy dependencies for functionality that can be implemented with NumPy or SciPy.
