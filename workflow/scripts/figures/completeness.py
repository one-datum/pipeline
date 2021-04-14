#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import fitsio
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, type=str)
parser.add_argument("-o", "--output", required=True, type=str)
parser.add_argument("-t", "--threshold", required=True, type=float)
args = parser.parse_args()

data = fitsio.read(args.input)

# Compute "detection" mask
det = np.isfinite(data["dr2_radial_velocity_error"])
det &= np.isfinite(data["rv_est_uncert"])
det &= data["dr2_rv_nb_transits"] >= 3
det &= data["rv_pval"] < args.threshold

# Plot the contour levels
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
x = np.log10(data["sim_semiamp"])
y = np.log10(data["rv_est_uncert"])
denom, bx, by = np.histogram2d(x, y, (15, 12))
num, _, _ = np.histogram2d(x[det], y[det], (bx, by))
c = ax.contourf(
    10 ** bx[:-1],
    10 ** by[:-1],
    100 * (num / denom).T,
    levels=np.linspace(0, 100, 9),
)
fig.colorbar(c, label="completeness [%]")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("per-transit uncertainty [km/s]")
ax.set_xlabel("K [km/s]")
fig.savefig(args.output, bbox_inches="tight", dpi=150)
fig.savefig(args.output.replace(".png", ".pdf"), bbox_inches="tight")
