#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter


def interpolate(X, Y, Z, x0, y0):
    x = x0
    Z1 = np.zeros_like(Z)
    for n in range(len(y0)):
        x_ref = X[:, n]
        y_ref = Z[:, n]
        mask = np.isfinite(x_ref) & np.isfinite(y_ref)
        Z1[:, n] = interp1d(
            x_ref[mask], y_ref[mask], fill_value="extrapolate"
        )(x)

    x = y0
    Z2 = np.zeros_like(Z)
    for n in range(len(mag_bin_centers)):
        x_ref = Y[n, :]
        y_ref = Z[n, :]
        mask = np.isfinite(x_ref) & np.isfinite(y_ref)
        Z2[n, :] = interp1d(
            x_ref[mask], y_ref[mask], fill_value="extrapolate"
        )(x)

    return gaussian_filter(0.5 * (Z1 + Z2), (0.5, 0.5))


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
        intercept = f[1].data
        slope = f[2].data
        count = f[3].data
        eff_mag = f[4].data
        eff_color = f[5].data
        # eff_log_plx = f[6].data
        mag_bins = f[7].data
        color_bins = f[8].data

    # Mask bins with a small number of targets
    mask = count > 100
    intercept[~mask] = np.nan
    slope[~mask] = np.nan

    # Compute the evenly spaced bin centers
    mag_bin_centers = 0.5 * (mag_bins[1:] + mag_bins[:-1])
    color_bin_centers = 0.5 * (color_bins[1:] + color_bins[:-1])

    # Interpolate horizontally and vertically
    intercept_full = interpolate(
        eff_mag, eff_color, intercept, mag_bin_centers, color_bin_centers
    )
    slope_full = interpolate(
        eff_mag, eff_color, slope, mag_bin_centers, color_bin_centers
    )

    # Estimate the scatter in this model
    intercept_scatter = np.sqrt(
        np.nanmedian((intercept - intercept_full) ** 2)
    )
    slope_scatter = np.sqrt(np.nanmedian((slope - slope_full) ** 2))

    # Save the results
    hdr["int_scat"] = intercept_scatter
    hdr["slp_scat"] = slope_scatter
    fits.HDUList(
        [
            fits.PrimaryHDU(header=hdr),
            fits.ImageHDU(mag_bins, name="mag bins"),
            fits.ImageHDU(color_bins, name="color bins"),
            fits.ImageHDU(intercept_full),
            fits.ImageHDU(slope_full),
            fits.ImageHDU(mask.astype(np.int32)),
            fits.ImageHDU(count),
        ]
    ).writeto(args.output, overwrite=True)
