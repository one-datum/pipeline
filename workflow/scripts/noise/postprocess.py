#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits
from scipy.interpolate import UnivariateSpline, interp1d


def interpolate(X, Y, Z, x0, y0):
    x = x0
    Z1 = np.zeros_like(Z)
    for n in range(len(y0)):
        x_ref = X[:, n]
        y_ref = Z[:, n]
        mask = np.isfinite(y_ref) & np.isfinite(y_ref)

        inds = np.arange(len(y_ref))
        resid = np.abs(np.diff(y_ref[mask]) / np.diff(x_ref[mask]))
        sigma = np.sqrt(np.nanmedian(resid ** 2))
        y_ref[inds[mask][:-1][resid > 3 * sigma]] = np.nan

        mask = np.isfinite(y_ref) & np.isfinite(y_ref)
        Z1[:, n] = interp1d(
            x_ref[mask], y_ref[mask], fill_value="extrapolate"
        )(x)

    x = y0
    Z2 = np.zeros_like(Z)
    for n in range(len(x0)):
        x_ref = Y[n, :]
        y_ref = Z[n, :]
        mask = np.isfinite(x_ref) & np.isfinite(y_ref)

        inds = np.arange(len(y_ref))
        resid = np.abs(np.diff(y_ref[mask]) / np.diff(x_ref[mask]))
        sigma = np.sqrt(np.nanmedian(resid ** 2))
        y_ref[inds[mask][:-1][resid > 3 * sigma]] = np.nan

        mask = np.isfinite(y_ref) & np.isfinite(y_ref)
        Z2[n, :] = interp1d(
            x_ref[mask], y_ref[mask], fill_value="extrapolate"
        )(x)

    model = 0.5 * (Z1 + Z2)

    # Handle edge effects
    model[:, 0] = UnivariateSpline(x0, model[:, 0])(x0)
    model[:, -1] = UnivariateSpline(x0, model[:, -1])(x0)
    model[0] = UnivariateSpline(y0, model[0])(y0)
    model[-1] = UnivariateSpline(y0, model[-1])(y0)

    return model


def postprocess(input, output):
    # Load the data file
    with fits.open(input) as f:
        hdr = f[0].header
        log_df = f[1].data
        loc = f[2].data
        log_scale = f[3].data
        count = f[4].data
        eff_mag = f[5].data
        eff_color = f[6].data
        mag_bins = f[7].data
        color_bins = f[8].data

    # Mask bins with few points
    log_df[count < 100] = np.nan
    loc[count < 100] = np.nan
    log_scale[count < 100] = np.nan

    # Compute the evenly spaced bin centers
    mag_bin_centers = 0.5 * (mag_bins[1:] + mag_bins[:-1])
    color_bin_centers = 0.5 * (color_bins[1:] + color_bins[:-1])

    # Interpolate horizontally and vertically
    log_df = interpolate(
        eff_mag, eff_color, log_df, mag_bin_centers, color_bin_centers
    )
    loc = interpolate(
        eff_mag, eff_color, loc, mag_bin_centers, color_bin_centers
    )
    log_scale = interpolate(
        eff_mag, eff_color, log_scale, mag_bin_centers, color_bin_centers
    )

    # Save the results
    fits.HDUList(
        [
            fits.PrimaryHDU(header=hdr),
            fits.ImageHDU(mag_bins, name="mag bins"),
            fits.ImageHDU(color_bins, name="color bins"),
            fits.ImageHDU(log_df),
            fits.ImageHDU(loc),
            fits.ImageHDU(log_scale),
            fits.ImageHDU(count),
        ]
    ).writeto(output, overwrite=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    args = parser.parse_args()

    postprocess(args.input, args.output)
