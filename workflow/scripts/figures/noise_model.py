#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter


def make_figure(mask, model, levels, output, norm=None):
    plt.figure(figsize=(7, 6))
    mappable = plt.contourf(
        color_bin_centers,
        mag_bin_centers,
        model,
        levels=levels,
        norm=norm,
    )

    # Overplot the mask
    color_array = np.zeros((2, 4))
    color_array[:, -1] = [1.0, 0]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        name="shading", colors=color_array
    )
    plt.contourf(
        color_bin_centers,
        mag_bin_centers,
        1.5 * mask,
        levels=1,
        cmap=cmap,
        vmin=0,
        vmax=1,
    )

    if norm is None:
        plt.colorbar(mappable, label=r"slope [km/s/ln(arcsec)]")
    else:
        sm = plt.cm.ScalarMappable(norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, label=r"per-transit uncertainty [km/s]")
        cbar.ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

    plt.ylim(plt.ylim()[::-1])
    plt.ylabel("$m_\mathrm{G}$")
    plt.xlabel("$G_\mathrm{BP}-G_\mathrm{RP}$")

    plt.savefig(output, bbox_inches="tight")


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, type=str)
parser.add_argument("-o", "--output", required=True, type=str)
args = parser.parse_args()

with fits.open(args.input) as f:
    mag_bins = f[1].data
    color_bins = f[2].data
    log_df = f[3].data
    loc = f[4].data
    log_scale = f[5].data
    count = f[6].data

mask = count > 1

color_bin_centers = 0.5 * (color_bins[1:] + color_bins[:-1])
mag_bin_centers = 0.5 * (mag_bins[1:] + mag_bins[:-1])

sigma_rv = np.exp(loc)
levels = np.exp(np.linspace(np.nanmin(loc[mask]), np.nanmax(loc[mask]), 25))
sigma_rv = np.clip(sigma_rv, levels[0] + 1e-5, levels[-1] - 1e-5)
# sigma_rv[sigma_rv >= levels[-1]] = levels[-1] - 1e-5

make_figure(
    mask,
    sigma_rv,
    levels,
    args.output,
    norm=mpl.colors.LogNorm(vmin=levels[0], vmax=levels[-1]),
)

# name, ext = os.path.splitext(args.output)
# make_figure(
#     slope,
#     np.linspace(slope[mask == 1].min(), slope[mask == 1].max(), 25),
#     f"{name}_slope{ext}"
#     # norm=mpl.colors.LogNorm(vmin=levels[0], vmax=levels[-1]),
# )
