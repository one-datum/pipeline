#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("--color-smooth", required=True, type=float)
    parser.add_argument("--mag-smooth", required=True, type=float)
    args = parser.parse_args()

    # Load the data file
    with fits.open(args.input) as f:
        hdr = f[0].header
        mu = f[1].data
        sigma = f[2].data
        count = f[3].data

    # Compute the bin coordinates from the header specification
    color_bins = np.linspace(
        hdr["MIN_COL"], hdr["MAX_COL"], hdr["NUM_COL"] + 1
    )
    mag_bins = np.linspace(hdr["MIN_MAG"], hdr["MAX_MAG"], hdr["NUM_MAG"] + 1)
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
    dc = args.color_smooth / (color_bins[1] - color_bins[0])
    dm = args.mag_smooth / (mag_bins[1] - mag_bins[0])
    mu_smooth = gaussian_filter(mu_full, (dm, dc))

    # Update this so that only the out of bounds parts are smoothed
    valid = np.isfinite(mu_base)
    mu_smooth[valid] = mu_base[valid]

    # Save the results
    hdr["col_smth"] = args.color_smooth
    hdr["mag_smth"] = args.mag_smooth
    fits.HDUList(
        [
            fits.PrimaryHDU(header=hdr),
            fits.ImageHDU(mu_smooth),
            fits.ImageHDU(valid.astype(np.int32)),
            fits.ImageHDU(count),
            fits.ImageHDU(mu),
            fits.ImageHDU(sigma),
        ]
    ).writeto(args.output, overwrite=True)
