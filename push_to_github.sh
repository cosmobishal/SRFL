#!/usr/bin/env bash
# =============================================================
# push_to_github.sh
# Run this script from the reconstructed repository root to
# apply all changes and push them to GitHub.
#
# Usage:
#   chmod +x push_to_github.sh
#   ./push_to_github.sh
# =============================================================

set -e

echo "=== SRFL Repository Reconstruction — Push Script ==="
echo ""

# Confirm git remote
REMOTE=$(git remote get-url origin 2>/dev/null || echo "NONE")
echo "Remote: $REMOTE"
echo ""

# Stage all changes
git add -A

# Show what is being committed
echo "Files to be committed:"
git status --short
echo ""

git commit -m "refactor: restructure repository to proper src-layout

- Move core library into src/srfl/ (kernel, field, defects, swarm, action, multiscale, cli)
- Move experiments into experiments/ (run_step, run_oscillatory, run_all)
- Move scripts into scripts/ (generate_figures)
- Move tests into tests/ (test_kernel, test_field, test_defects, test_swarm, test_action, test_multiscale)
- Move paper into paper/ (srfl_paper.tex)
- Move figures into figures/ (all PNG outputs)
- Add pyproject.toml with setuptools build system config
- Add .gitignore covering Python, LaTeX, Jupyter artefacts
- Add LICENSE (MIT)
- Add .github/workflows/ci.yml (test + lint + figures CI)
- Add notebooks/srfl_demo.ipynb (interactive walkthrough)
- Fix run_all.py sys.path to find both src/srfl and experiments/
- Fix setup.py: add url, extras_require, expanded classifiers
- Fix README.md: correct image paths to figures/ directory
- Fix run_tests.py: correct sys.path to src/
- All 51 tests pass"

echo ""
echo "Pushing to GitHub..."
git push origin main

echo ""
echo "=== Done! Visit: https://github.com/cosmobishal/SRFL ==="
