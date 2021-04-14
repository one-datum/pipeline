#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Union

import fitsio
import numpy as np
from astropy.io import fits
from numpy.lib.recfunctions import append_fields
from scipy.interpolate import RegularGridInterpolator


def get_uncertainty_model(
    filename: str,
    *,
    bounds_error: bool = False,
    fill_value: float = None,
) -> RegularGridInterpolator:
    with fits.open(filename) as f:
        hdr = f[0].header
        mu = f[1].data

    color_bins = np.linspace(
        hdr["MIN_COL"], hdr["MAX_COL"], hdr["NUM_COL"] + 1
    )
    mag_bins = np.linspace(hdr["MIN_MAG"], hdr["MAX_MAG"], hdr["NUM_MAG"] + 1)
    return RegularGridInterpolator(
        [
            0.5 * (mag_bins[1:] + mag_bins[:-1]),
            0.5 * (color_bins[1:] + color_bins[:-1]),
        ],
        mu,
        bounds_error=bounds_error,
        fill_value=fill_value,
    )


def load_data(
    data_path: str,
    *,
    rows: Union[Iterable[int], slice] = slice(None),
    cols: Iterable[str] = [
        "source_id",
        "ra",
        "dec",
        "parallax",
        "bp_rp",
        "phot_g_mean_mag",
        "dr2_rv_nb_transits",
        "dr2_radial_velocity_error",
    ],
    ext: int = 1,
) -> np.recarray:
    with fitsio.FITS(data_path) as f:
        return f[ext][cols][rows]


def compute_ln_sigma(
    data: np.recarray, *, step: int = 500, filename: str = None
) -> np.ndarray:
    noise_model = get_uncertainty_model(filename)
    ln_sigma = np.empty(len(data), dtype=np.float32)
    ln_sigma[:] = np.nan
    for n in range(0, len(data), step):
        mask = np.isfinite(
            data["phot_g_mean_mag"][n : n + step]
        ) & np.isfinite(data["bp_rp"][n : n + step])
        ln_sigma[n : n + step][mask] = noise_model(
            np.concatenate(
                (
                    data["phot_g_mean_mag"][n : n + step][mask, None],
                    data["bp_rp"][n : n + step][mask, None],
                ),
                axis=1,
            )
        )
    return ln_sigma


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-n", "--noise-model", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    args = parser.parse_args()

    print("Loading data...")
    data = load_data(args.input)

    print("Estimating sigma...")
    ln_sigma = compute_ln_sigma(data, filename=args.noise_model)
    data = append_fields(data, ["rv_est_uncert"], [np.exp(ln_sigma)])
    fitsio.write(args.output, data, clobber=True)
