#!/usr/bin/env bash
set -e

. /opt/conda/etc/profile.d/conda.sh
conda activate one-datum

# First, install the package
python -m pip install -e .

cd paper/figures

# Save the git commit hash
echo "\\newcommand{\\githash}{$(git rev-parse HEAD)}" > githash.tex

# Generate the figures
python demo_figure.py
