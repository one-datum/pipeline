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

# Plot the recovered semi-amplitude
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
c = ax.scatter(
    data["sim_semiamp"][det],
    data["rv_semiamp_p50"][det],
    s=3,
    c=data["sim_ecc"][det],
    alpha=0.5,
    vmin=0,
    vmax=1,
    edgecolor="none",
    rasterized=True,
    cmap="YlGn_r",
)
fig.colorbar(c, label="simulated eccentricity")
ax.set_xscale("log")
ax.set_yscale("log")
rng = ax.get_xlim()
ax.plot(rng, rng, "k", lw=0.5)
ax.set_xlim(rng)
ax.set_ylim(rng)
ax.set_xlabel("simulated semi-amplitude [km/s]")
ax.set_ylabel("median recovered semi-amplitude [km/s]")
fig.savefig(args.output, bbox_inches="tight")
