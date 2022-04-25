#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Tuple, Union

import fitsio
import numpy as np
import scipy.stats
from astropy.io import fits
from numpy.lib.recfunctions import append_fields
from scipy.interpolate import RegularGridInterpolator


def get_uncertainty_model(
    filename: str,
    *,
    bounds_error: bool = False,
    fill_value: float = np.nan,
) -> Tuple[RegularGridInterpolator, RegularGridInterpolator]:
    with fits.open(filename) as f:
        hdr = f[0].header
        mu = f[1].data
        mask = f[2].data

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
    ), RegularGridInterpolator(
        [
            0.5 * (mag_bins[1:] + mag_bins[:-1]),
            0.5 * (color_bins[1:] + color_bins[:-1]),
        ],
        mask.astype(np.float32),
        bounds_error=False,
        fill_value=0.0,
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
    data: np.recarray, *, filename: str, step: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    noise_model, mask_model = get_uncertainty_model(filename)
    ln_sigma = np.empty(len(data), dtype=np.float32)
    ln_sigma[:] = np.nan
    confidence = np.zeros(len(data), dtype=np.float32)
    for n in range(0, len(data), step):
        mask = np.isfinite(
            data["phot_g_mean_mag"][n : n + step]
        ) & np.isfinite(data["bp_rp"][n : n + step])
        X = np.concatenate(
            (
                data["phot_g_mean_mag"][n : n + step][mask, None],
                data["bp_rp"][n : n + step][mask, None],
            ),
            axis=1,
        )
        ln_sigma[n : n + step][mask] = noise_model(X)
        confidence[n : n + step][mask] = mask_model(X)
    return ln_sigma, confidence


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
    ln_sigma, confidence = compute_ln_sigma(data, filename=args.noise_model)

    print("Computing p-values...")
    nb_transits = data["dr2_rv_nb_transits"].astype(np.int32)
    eps = data["dr2_radial_velocity_error"]
    sample_variance = 2 * nb_transits * (eps**2 - 0.11**2) / np.pi
    statistic = sample_variance * (nb_transits - 1)
    pval = 1 - scipy.stats.chi2(nb_transits - 1).cdf(
        statistic * np.exp(-2 * ln_sigma)
    ).astype(np.float32)

    data = append_fields(
        data,
        ["rv_est_uncert", "rv_unc_conf", "rv_pval"],
        [np.exp(ln_sigma), confidence, pval],
    )
    fitsio.write(args.output, data, clobber=True)
