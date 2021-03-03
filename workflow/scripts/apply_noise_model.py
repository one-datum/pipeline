#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Union

import fitsio
import numpy as np
from numpy.lib.recfunctions import append_fields

from one_datum import uncertainty


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


def compute_ln_sigma(data: np.recarray, *, step: int = 500) -> np.ndarray:
    noise_model = uncertainty.get_uncertainty_model()
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
    parser.add_argument("-o", "--output", required=True, type=str)
    args = parser.parse_args()

    print("Loading data...")
    data = load_data(args.input)

    print("Estimating sigma...")
    ln_sigma = compute_ln_sigma(data)
    data = append_fields(data, ["noise_ln_sigma"], [ln_sigma])
    fitsio.write(args.output, data, clobber=True)
