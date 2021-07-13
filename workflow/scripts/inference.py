#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
from multiprocessing import Pool
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import Tuple

import fitsio
import kepler
import numpy as np
import scipy.stats
from numpy.lib.recfunctions import append_fields

QUANTILES = np.array([0.05, 0.16, 0.5, 0.84, 0.95])


def precompute_model(
    max_nb_transits: int, *, num_samp: int = 50000, seed: int = 384820
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    random = np.random.default_rng(seed)

    # Simulate transit times by sampling from the baseline
    t = random.uniform(0, 668, (max_nb_transits, num_samp))

    # Sample many parameters from the prior
    log_period = random.uniform(np.log(1.0), np.log(800.0), num_samp)
    phase = random.uniform(-np.pi, np.pi, num_samp)
    log_semiamp = np.sort(
        random.uniform(np.log(0.1), np.log(1000.0), num_samp)
    )
    # ecc = random.beta(0.867, 3.03, num_samp)
    ecc = random.uniform(0.0, 0.9, num_samp)
    omega = random.uniform(-np.pi, np.pi, num_samp)

    # Compute the Keplerian model
    cosw = np.cos(omega)
    sinw = np.sin(omega)
    M = 2 * np.pi * t * np.exp(-log_period)[None, :] + phase[None, :]
    _, cosf, sinf = kepler.kepler(M, ecc[None, :] + np.zeros_like(M))
    rv_mod = np.exp(log_semiamp[None, :]) * (
        cosw[None, :] * (ecc[None, :] + cosf) - sinw[None, :] * sinf
    )

    lam = np.zeros((max_nb_transits + 1, num_samp), dtype=np.float32)
    for n in range(2, max_nb_transits + 1):
        m = rv_mod[: n + 1]
        lam[n] = np.sum((m - np.mean(m, axis=0)[None, :]) ** 2, axis=0)

    return log_semiamp.astype(np.float32), lam, random.normal(size=num_samp)


def _inner_process(
    rate_param: np.ndarray,
    eps: np.ndarray,
    ln_sigma: np.ndarray,
    nb_transits: np.ndarray,
    statistic: np.ndarray,
):
    # The Keplerian model
    ivar = np.exp(-2 * (ln_sigma[:, None] + eps[None, :]))
    target_lam = rate_param[nb_transits - 1] * ivar
    ncx2 = scipy.stats.ncx2(df=nb_transits[:, None], nc=target_lam)

    s2 = statistic[:, None] * ivar
    log_weight = ncx2.logpdf(s2)
    weights = np.exp(
        log_weight - log_weight.max(axis=1)[:, None], out=target_lam
    )

    # Compute the quantiles assuming that ln_semiamp is sorted
    cdf = np.cumsum(weights, axis=1, out=weights)
    return np.array(
        [
            np.clip(np.searchsorted(c, QUANTILES * c[-1]) - 1, 0, len(c) - 1)
            for c in cdf
        ]
    )


def process_shared(
    name1: str,
    shape1: Tuple[int],
    dtype1: np.dtype,
    name2: str,
    shape2: Tuple[int],
    dtype2: np.dtype,
    args: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    rate_buf = SharedMemory(name1)
    rate_param = np.ndarray(shape=shape1, dtype=dtype1, buffer=rate_buf.buf)
    eps_buf = SharedMemory(name2)
    eps = np.ndarray(shape=shape2, dtype=dtype2, buffer=eps_buf.buf)
    return _inner_process(rate_param, eps, *args)


if __name__ == "__main__":
    import argparse
    import time
    import tracemalloc

    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, type=str)
    parser.add_argument("--calib", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--block-size", default=10, type=int)
    args = parser.parse_args()

    block_size = int(args.block_size)

    print("Loading data...")
    data = fitsio.read(args.catalog)

    with open(args.calib, "r") as f:
        noise_err = np.mean(list(map(float, f.readlines())))

    # Compute the ingredients for probabilistic model
    ln_sigma = np.log(data["rv_est_uncert"])
    nb_transits = data["dr2_rv_nb_transits"].astype(np.int32)
    eps = data["dr2_radial_velocity_error"]
    sample_variance = 2 * nb_transits * (eps ** 2 - 0.11 ** 2) / np.pi
    statistic = (sample_variance * (nb_transits - 1)).astype(np.float32)
    pval = data["rv_pval"]
    valid = (
        (nb_transits >= 3)
        & (pval < 0.01)
        & np.isfinite(ln_sigma)
        & np.isfinite(eps)
    )
    ln_sigma = ln_sigma[valid]
    nb_transits = nb_transits[valid]
    statistic = statistic[valid]
    max_nb_transits = data["dr2_rv_nb_transits"].max()
    num_rows = len(statistic)

    # Simulate the RV error model
    print("Simulating model...")
    ln_semiamp, rate_param, eps = precompute_model(max_nb_transits)
    eps *= noise_err

    print("Processing shared...")
    tracemalloc.start()
    start_time = time.time()
    with SharedMemoryManager() as smm:
        shared_rate_param = smm.SharedMemory(rate_param.nbytes)  # type: ignore
        rate_array = np.ndarray(
            shape=rate_param.shape,
            dtype=rate_param.dtype,
            buffer=shared_rate_param.buf,
        )
        rate_array[:] = rate_param

        shared_eps = smm.SharedMemory(eps.nbytes)  # type: ignore
        eps_array = np.ndarray(
            shape=eps.shape,
            dtype=eps.dtype,
            buffer=shared_eps.buf,
        )
        eps_array[:] = eps

        func = partial(
            process_shared,
            shared_rate_param.name,
            rate_param.shape,
            rate_param.dtype,
            shared_eps.name,
            eps.shape,
            eps.dtype,
        )
        with Pool() as pool:
            results = list(
                pool.map(
                    func,
                    [
                        (
                            ln_sigma[n : n + block_size],
                            nb_transits[n : n + block_size],
                            statistic[n : n + block_size],
                        )
                        for n in range(0, num_rows, block_size)
                    ],
                )
            )
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage {current/1e6}MB; Peak: {peak/1e6}MB")
    print(f"Time elapsed: {time.time()-start_time:.2f}s")
    tracemalloc.stop()

    # Save the results
    inds = np.concatenate(results, axis=0)
    result = np.empty((len(data), len(QUANTILES)), dtype=np.float32)
    result[:] = np.nan
    result[valid] = ln_semiamp[inds]
    data = append_fields(
        data,
        ["rv_variance"] + [f"rv_semiamp_p{(100 * q):.0f}" for q in QUANTILES],
        [sample_variance]
        + [np.exp(result[:, q]) for q in range(len(QUANTILES))],
    )
    fitsio.write(args.output, data, clobber=True)
