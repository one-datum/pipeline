#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

if __name__ == "__main__":
    from custom_logger import setup_logger

    setup_logger(filename=snakemake.log[0])

    # Load the data file
    with fits.open(snakemake.input[0]) as f:
        hdr = f[0].header
        mu = f[1].data
        count = f[3].data

    # Compute the bin coordinates from the header specification
    color_bins = np.linspace(
        hdr["MIN_COL"], hdr["MAX_COL"], hdr["NUM_COL"] + 1
    )
    mag_bins = np.linspace(
        hdr["MIN_MAG"], hdr["MAX_MAG"], hdr["NUM_MAG"] + 1
    )
    color_bin_centers = 0.5 * (color_bins[1:] + color_bins[:-1])
    mag_bin_centers = 0.5 * (mag_bins[1:] + mag_bins[:-1])

    # Interpolate horizontally and vertically
    mu_base = np.mean(mu, axis=-1)
    mn, mx = (
        mu_base[np.isfinite(mu_base)].min(),
        mu_base[np.isfinite(mu_base)].max(),
    )

    mu_mean_x = np.copy(mu_base)
    for k in range(mu_mean_x.shape[0]):
        y = mu_mean_x[k]
        m = np.isfinite(y)
        mu_mean_x[k] = interp1d(
            color_bin_centers[m],
            y[m],
            kind="nearest",
            fill_value="extrapolate",
        )(color_bin_centers)

    mu_mean_y = np.copy(mu_base)
    for k in range(mu_mean_y.shape[1]):
        y = mu_mean_y[:, k]
        m = np.isfinite(y)
        mu_mean_y[:, k] = interp1d(
            mag_bin_centers[m],
            y[m],
            kind="nearest",
            fill_value="extrapolate",
        )(mag_bin_centers)

    # Take the mean of the interpolations; this shouldn't change the values
    # in bounds, but should somewhat smooth the out of bounds regions
    mu_full = 0.5 * (mu_mean_x + mu_mean_y)

    # Finally, smooth the model using a Gaussian filter
    dc = snakemake.config["noise"]["color_smoothing_scale"] / (
        color_bins[1] - color_bins[0]
    )
    dm = snakemake.config["noise"]["mag_smoothing_scale"] / (
        mag_bins[1] - mag_bins[0]
    )
    mu_smooth = gaussian_filter(mu_full, (dm, dc))

    # Save the results
    hdr["col_smth"] = snakemake.config["noise"]["color_smoothing_scale"]
    hdr["mag_smth"] = snakemake.config["noise"]["mag_smoothing_scale"]
    fits.HDUList(
        [
            fits.PrimaryHDU(header=hdr),
            fits.ImageHDU(mu_smooth),
            fits.ImageHDU(np.isfinite(mu_base).astype(np.int32)),
            fits.ImageHDU(count),
        ]
    ).writeto(snakemake.output[0], overwrite=True)
