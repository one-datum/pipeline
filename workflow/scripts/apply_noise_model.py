#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
from multiprocessing import Pool
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import Iterable, Tuple, Union

import fitsio
import kepler
import numpy as np
import scipy.stats
from numpy.lib.recfunctions import append_fields

from one_datum import uncertainty

QUANTILES = np.array([0.05, 0.16, 0.5, 0.84, 0.95])


def load_data(
    data_path: str,
    *,
    rows: Union[Iterable[int], slice] = slice(None),
    cols: Iterable[str] = [
        "source_id",
        "ra",
        "dec",
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


def precompute_model(
    max_nb_transits: int, *, num_samp: int = 50000, seed: int = 384820
) -> Tuple[np.ndarray, np.ndarray]:
    random = np.random.default_rng(seed)

    # Simulate transit times by sampling from the baseline
    t = random.uniform(0, 668, (max_nb_transits, num_samp))

    # Sample many parameters from the prior
    log_period = random.uniform(np.log(1.0), np.log(800.0), num_samp)
    phase = random.uniform(-np.pi, np.pi, num_samp)
    log_semiamp = np.sort(random.uniform(np.log(0.1), np.log(500.0), num_samp))
    ecc = random.beta(0.867, 3.03, num_samp)
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

    return log_semiamp.astype(np.float32), lam


def _inner_process(
    rate_param: np.ndarray,
    ln_sigma: np.ndarray,
    nb_transits: np.ndarray,
    statistic: np.ndarray,
):
    # The Keplerian model
    ivar = np.exp(-2 * ln_sigma)
    target_lam = rate_param[nb_transits - 1] * ivar[:, None]
    ncx2 = scipy.stats.ncx2(df=nb_transits[:, None], nc=target_lam)

    s2 = np.multiply(statistic, ivar, out=ivar)
    log_weight = ncx2.logpdf(s2[:, None])
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


def process(
    rate_param: np.ndarray,
    args: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    ln_sigma, nb_transits, statistic = args
    return _inner_process(rate_param, *args)


def process_shared(
    name: str,
    shape: Tuple[int],
    dtype: np.dtype,
    args: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    rate_buf = SharedMemory(name)
    rate_param = np.ndarray(shape=shape, dtype=dtype, buffer=rate_buf.buf)
    return _inner_process(rate_param, *args)


if __name__ == "__main__":
    import argparse
    import time
    import tracemalloc

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    args = parser.parse_args()

    block_size = 10
    filename = args.input

    print("Loading data...")
    data = load_data(filename)
    num_rows = len(data)
    max_nb_transits = data["dr2_rv_nb_transits"].max()

    # Simulate the RV error model
    print("Simulating model...")
    ln_semiamp, rate_param = precompute_model(data["dr2_rv_nb_transits"].max())

    # Compute the ingredients for probabilistic model
    ln_sigma = compute_ln_sigma(data)
    nb_transits = data["dr2_rv_nb_transits"].astype(np.int32)
    eps = data["dr2_radial_velocity_error"]
    sample_variance = 2 * nb_transits * (eps ** 2 - 0.11 ** 2) / np.pi
    statistic = (sample_variance * (nb_transits - 1)).astype(np.float32)

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

        func = partial(
            process_shared,
            shared_rate_param.name,
            rate_param.shape,
            rate_param.dtype,
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
                        for n in range(0, len(data), block_size)
                    ],
                )
            )
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage {current/1e6}MB; Peak: {peak/1e6}MB")
    print(f"Time elapsed: {time.time()-start_time:.2f}s")
    tracemalloc.stop()

    # Save the results
    inds = np.concatenate(results, axis=0)
    ln_semiamp = ln_semiamp[inds]
    data = append_fields(
        data,
        ["noise_ln_sigma"]
        + [f"noise_semiamp_p{(100 * q):.0f}" for q in QUANTILES],
        [ln_sigma] + [np.exp(ln_semiamp[:, q]) for q in range(len(QUANTILES))],
    )
    fitsio.write(args.output, data, clobber=True)
