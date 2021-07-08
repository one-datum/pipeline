#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
from multiprocessing import Pool
from typing import Tuple

import fitsio
import numpy as np
import scipy.stats


def estimate_sigma(
    seed: int, args: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> float:
    nb_transits, statistic, ln_sigma = args
    dist = scipy.stats.chi2(nb_transits - 1)
    eps = np.random.default_rng(seed).normal(size=(100, len(nb_transits)))

    def eval_ks(sigma):
        pval = np.mean(
            1 - dist.cdf(statistic * np.exp(-2 * (ln_sigma + sigma * eps))),
            axis=0,
        )
        sorted_pval = np.sort(pval[np.isfinite(pval) & (pval > 0.5)])
        return np.max(
            np.abs(np.linspace(0.5, 1.0, len(sorted_pval)) - sorted_pval)
        )

    sigmas = np.linspace(0.16, 0.22, 20)
    ks = np.empty_like(sigmas)
    for n, sigma in enumerate(sigmas):
        ks[n] = eval_ks(sigma)

    return sigmas[np.argmin(ks)]


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-c", "--config", required=True, type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    print("Loading data...")
    data = fitsio.read(args.input)

    # Extract the relevant columns
    nb_transits = data["dr2_rv_nb_transits"].astype(np.int32)
    eps = data["dr2_radial_velocity_error"]
    sample_variance = 2 * nb_transits * (eps ** 2 - 0.11 ** 2) / np.pi
    statistic = sample_variance * (nb_transits - 1)
    ln_sigma = np.log(data["rv_est_uncert"])

    # select random subsets of the data
    rng = np.random.default_rng(config["seed"])
    inds = [
        rng.integers(len(data), size=10000) for n in range(config["num_calib"])
    ]

    with Pool() as pool:
        results = list(
            pool.map(
                partial(estimate_sigma, config["seed"]),
                ((nb_transits[i], statistic[i], ln_sigma[i]) for i in inds),
            )
        )

    with open(args.output, "w") as f:
        f.write("\n".join(map("{0}".format, results)))
