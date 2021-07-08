#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
from multiprocessing import Pool
from typing import Tuple

import fitsio
import numpy as np
import scipy.stats
from numpy.lib.recfunctions import append_fields


def compute_pval(
    eps: np.ndarray,
    args: Tuple[np.ndarray, np.ndarray, np.ndarray],
):
    nb_transits, statistic, ln_sigma = args
    dist = scipy.stats.chi2(nb_transits - 1)
    return np.mean(
        1
        - dist.cdf(
            statistic * np.exp(-2 * (ln_sigma + eps[:, : len(ln_sigma)]))
        ),
        axis=0,
    )


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, type=str)
    parser.add_argument("--calib", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    print("Loading data...")
    data = fitsio.read(args.catalog)

    with open(args.calib, "r") as f:
        noise_err = np.mean(list(map(float, f.readlines())))
    print(noise_err)

    # Extract the relevant columns
    nb_transits = data["dr2_rv_nb_transits"].astype(np.int32)
    eps = data["dr2_radial_velocity_error"]
    sample_variance = 2 * nb_transits * (eps ** 2 - 0.11 ** 2) / np.pi
    statistic = sample_variance * (nb_transits - 1)
    ln_sigma = np.log(data["rv_est_uncert"])

    num_pval = int(config["num_pval"])
    eps = noise_err * np.random.default_rng(config["seed"]).normal(
        size=(100, num_pval)
    )
    slices = [
        slice(n, n + num_pval if n + num_pval <= len(data) else None)
        for n in range(0, len(data), num_pval)
    ]

    results = np.empty_like(ln_sigma)
    with Pool() as pool:
        for slc, pval in zip(
            slices,
            pool.map(
                partial(compute_pval, eps),
                ((nb_transits[s], statistic[s], ln_sigma[s]) for s in slices),
            ),
        ):
            results[slc] = pval

    data = append_fields(data, ["rv_pval"], [results])
    fitsio.write(args.output, data, clobber=True)
