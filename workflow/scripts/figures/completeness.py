#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, type=str)
parser.add_argument("-o", "--output", required=True, type=str)
parser.add_argument("-t", "--threshold", required=True, type=float)
args = parser.parse_args()

with fits.open(args.input) as f:
    data = f[1].data

# Compute "detection" mask
det = np.isfinite(data["radial_velocity_error"])
det &= np.isfinite(data["rv_ln_uncert"])
det &= data["rv_nb_transits"] >= 3
det &= data["rv_pval"] < args.threshold

# Plot the contour levels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

x = np.log10(data["sim_semiamp"])
y = np.log10(np.exp(data["rv_ln_uncert"]))
denom, bx, by = np.histogram2d(x, y, (15, 12))
num, _, _ = np.histogram2d(x[det], y[det], (bx, by))
c = ax1.contourf(
    10 ** by[:-1],
    10 ** bx[:-1],
    100 * (num / denom),
    levels=np.linspace(0, 100, 9),
)
ax1.plot(10 ** by[:-1], 10 ** by[:-1], "k")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("per-transit uncertainty [km/s]")
ax1.set_ylabel("semi-amplitude [km/s]")

x = np.log10(data["sim_semiamp"])
y = np.log10(data["rv_nb_transits"])
denom, bx, by = np.histogram2d(
    x, y, (15, 12), range=[None, (np.log10(2), np.log10(80))]
)
num, _, _ = np.histogram2d(x[det], y[det], (bx, by))
num[denom == 0] = 0
denom[denom == 0] = 1.0
c = ax2.contourf(
    10 ** by[:-1],
    10 ** bx[:-1],
    100 * (num / denom),
    levels=np.linspace(0, 100, 9),
)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("number of transits")
# ax2.set_ylabel("K [km/s]")

fig.subplots_adjust(right=0.85, wspace=0.1)
cbar_ax = fig.add_axes([0.854, 0.125, 0.02, 0.75])
fig.colorbar(c, label="completeness [%]", cax=cbar_ax)

fig.savefig(args.output, bbox_inches="tight")
