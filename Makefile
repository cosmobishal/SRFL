.PHONY: test build examples benchmark

test:
	PYTHONPATH=src pytest

build:
	python -m pip wheel . --no-build-isolation --no-deps -w dist

examples:
	PYTHONPATH=src python examples/step.py
	PYTHONPATH=src python examples/oscillatory.py

benchmark:
	PYTHONPATH=src python benchmarks/benchmark_targets.py
