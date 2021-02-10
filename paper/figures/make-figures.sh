#!/usr/bin/env bash
set -e

# First, install the package
python -m pip install -e .

cd paper/figures

# Save the git commit hash
echo "\\newcommand{\\githash}{$(git rev-parse HEAD)}" > githash.tex

# Generate the figures
python demo_figure.py
