#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fitsio
import kepler
import numpy as np
import scipy.stats
import yaml
from scipy.special import gamma


def simulate_nb_transits(random, N, x0, alpha):
    u = random.uniform(0, 1, N)
    x = np.empty_like(u)

    alpha = 6.0
    x0 = 12.0
    A = gamma(alpha) / (x0 * (gamma(alpha) + gamma(alpha - 1)))

    low = u < A * x0
    x[low] = u[low] / A
    x[~low] = x0 * ((1 - alpha) * (u[~low] - A * x0) / (A * x0) + 1) ** (
        1 / (1 - alpha)
    )
    x = np.ceil(x)
    return x.astype(np.int32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-c", "--config", required=True, type=str)
    args = parser.parse_args()

    data = fitsio.read(args.input)
    max_rv_err = data["radial_velocity_error"].max()

    with open(args.config, "r") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    print("Simulating model...")
    random = np.random.default_rng(int(config["seed"]))
    num_sims = int(config["num_sims"])

    # Simulate number of transits and baselines using the empirical distribution
    inds = random.integers(len(data), size=num_sims)
    nb_transits = data["rv_nb_transits"][inds].astype(np.int32)
    baselines = data["rv_time_duration"][inds]

    # Parameters
    ln_sigma_err = 0.05
    ln_sigma = random.uniform(
        np.log(config["rv_est_uncert"]["min"]),
        np.log(config["rv_est_uncert"]["max"]),
        num_sims,
    )
    del_ln_sigma = ln_sigma_err * random.standard_normal(num_sims)
    period = np.exp(
        random.uniform(
            np.log(config["period"]["min"]),
            np.log(config["period"]["max"]),
            num_sims,
        )
    )
    semiamp = np.exp(
        random.uniform(
            np.log(config["semiamp"]["min"]),
            np.log(config["semiamp"]["max"]),
            num_sims,
        )
    )
    ecc = random.beta(
        config["ecc"]["beta_a"], config["ecc"]["beta_b"], num_sims
    )
    omega = np.exp(random.uniform(-np.pi, np.pi, num_sims))
    phase = np.exp(random.uniform(0, 2 * np.pi, num_sims))
    cosw = np.cos(omega)
    sinw = np.sin(omega)

    norm = np.random.default_rng(55).standard_normal(50)

    # Simulate the RV error
    rv_err = np.full(num_sims, np.nan)
    pval = np.full(num_sims, np.nan)
    pval_err = np.full(num_sims, np.nan)
    for n in range(num_sims):
        t = random.uniform(0, baselines[n], nb_transits[n])
        M = 2 * np.pi * t / period[n] + phase[n]
        _, cosf, sinf = kepler.kepler(M, ecc[n] + np.zeros_like(M))
        rv_mod = semiamp[n] * (cosw[n] * (ecc[n] + cosf) - sinw[n] * sinf)
        rv_mod += np.exp(
            ln_sigma[n] + del_ln_sigma[n]
        ) * random.standard_normal(nb_transits[n])
        sample_variance = np.var(rv_mod, ddof=1)
        rv_err[n] = np.sqrt(
            0.5 * np.pi * sample_variance / nb_transits[n] + 0.11**2
        )
        if rv_err[n] > max_rv_err:
            rv_err[n] = np.nan

        statistic = sample_variance * (nb_transits[n] - 1)
        arg = ln_sigma[n] + ln_sigma_err * norm
        arg = 1 - scipy.stats.chi2(nb_transits[n] - 1).cdf(
            statistic * np.exp(-2 * arg)
        )
        pval[n] = np.mean(arg)
        pval_err[n] = np.std(arg)

    data = np.empty(
        num_sims,
        dtype=[
            ("rv_ln_uncert", np.float32),
            ("rv_ln_uncert_err", np.float32),
            ("rv_pval", np.float32),
            ("rv_pval_err", np.float32),
            ("rv_nb_transits", np.int32),
            ("radial_velocity_error", np.float32),
            ("sim_period", np.float32),
            ("sim_semiamp", np.float32),
            ("sim_ecc", np.float32),
            ("sim_omega", np.float32),
            ("sim_phase", np.float32),
        ],
    )
    data["rv_ln_uncert"] = ln_sigma
    data["rv_ln_uncert_err"][:] = ln_sigma_err
    data["rv_pval"] = pval
    data["rv_pval_err"] = pval_err
    data["rv_nb_transits"] = nb_transits
    data["radial_velocity_error"] = rv_err
    data["sim_period"] = period
    data["sim_semiamp"] = semiamp
    data["sim_ecc"] = ecc
    data["sim_omega"] = omega
    data["sim_phase"] = phase
    fitsio.write(args.output, data, clobber=True)
