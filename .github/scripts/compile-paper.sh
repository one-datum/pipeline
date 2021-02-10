#!/usr/bin/env bash
set -e

. /opt/conda/etc/profile.d/conda.sh
conda activate one-datum

cd paper
make
