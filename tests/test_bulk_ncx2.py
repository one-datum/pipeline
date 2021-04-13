# -*- coding: utf-8 -*-

import kepler
import numpy as np

from one_datum import ncx2


def precompute_model(
    max_nb_transits: int, *, num_samp: int = 50000, seed: int = 384820
):
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


def test_bulk_ncx2():
    num_data = 1000
    max_nb_transits = 100
    _, lam = precompute_model(max_nb_transits)

    ln_sigma = np.random.randn(num_data)
    nb_transits = np.random.randint(5, max_nb_transits, num_data).astype(
        np.int32
    )
    statistic = (nb_transits - 1) * np.exp(2 * np.random.randn(num_data))

    print(
        ncx2.evaluate_model(
            np.array([0.05, 0.16, 0.5, 0.84, 0.95]),
            lam,
            ln_sigma,
            nb_transits,
            statistic,
        )
    )
    assert 0
