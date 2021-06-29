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

sigma = data["rv_est_uncert"]
nb_transits = data["dr2_rv_nb_transits"].astype(np.int32)
m = np.isfinite(sigma) & (nb_transits >= 3)
sigma = sigma[m]
nb_transits = nb_transits[m]
pval = data["rv_pval"][m]
color = data["bp_rp"][m]
mag = data["phot_g_mean_mag"][m]
dm = 5 * np.log10(1000 / data["parallax"][m]) - 5

rng = (color < 2.75) & (color > 0.15)
rng &= (mag - dm > -4.0) & (mag - dm < 10.0)
denom, bx, by = np.histogram2d(color[rng], mag[rng] - dm[rng], bins=(80, 95))

binary = pval < args.threshold
num, bx, by = np.histogram2d(
    color[binary], mag[binary] - dm[binary], bins=(bx, by)
)

plt.figure(figsize=(7, 6))
plt.contour(
    0.5 * (bx[1:] + bx[:-1]),
    0.5 * (by[1:] + by[:-1]),
    denom.T,
    cmap="Greys",
    linewidths=[1],
)

v = np.zeros_like(num, dtype=np.float64)
v[:] = np.nan
m = denom > 50
v[m] = num[m] / denom[m]
plt.pcolor(
    bx,
    by,
    v.T,
    cmap="Reds",
    alpha=1.0,
    edgecolor="none",
    vmin=0,
    vmax=1,
    rasterized=True,
)

plt.ylim(plt.ylim()[::-1])
plt.colorbar(label="non-null fraction")
plt.xlabel("$G_\mathrm{BP} - G_\mathrm{RP}$")
plt.ylabel("$m_\mathrm{G} +$ DM")

plt.savefig(args.output, bbox_inches="tight")
