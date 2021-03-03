# -*- coding: utf-8 -*-

__all__ = ["get_filename", "get_uncertainty_model"]

from typing import Optional

import numpy as np
import pkg_resources
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator


def get_filename() -> str:
    return pkg_resources.resource_filename(
        __name__, "data/edr3-noise-model.fits"
    )


def get_uncertainty_model(
    *, bounds_error: bool = False, fill_value: Optional[float] = None
) -> RegularGridInterpolator:
    filename = get_filename()

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
