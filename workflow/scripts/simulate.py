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
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-c", "--config", required=True, type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    print("Simulating model...")
    random = np.random.default_rng(int(config["seed"]))
    num_sims = int(config["num_sims"])
    max_rv_err = float(config["max_rv_error"])

    # Simulate number of transits using approximate power law model
    nb_transits = np.floor(
        np.exp(random.uniform(0.0, np.log(200), num_sims))
    ).astype(np.int32)
    # nb_transits = simulate_nb_transits(
    #     random,
    #     num_sims,
    #     config["nb_transits"]["n_break"],
    #     config["nb_transits"]["power"],
    # )

    # Parameters
    rv_est_uncert = np.exp(
        random.uniform(
            np.log(config["rv_est_uncert"]["min"]),
            np.log(config["rv_est_uncert"]["max"]),
            num_sims,
        )
    )
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

    # Simulate the RV error
    rv_err = np.full(num_sims, np.nan)
    for n in range(num_sims):
        t = random.uniform(
            config["time"]["min"], config["time"]["max"], nb_transits[n]
        )
        M = 2 * np.pi * t / period[n] + phase[n]
        _, cosf, sinf = kepler.kepler(M, ecc[n] + np.zeros_like(M))
        rv_mod = semiamp[n] * (cosw[n] * (ecc[n] + cosf) - sinw[n] * sinf)
        rv_mod += rv_est_uncert[n] * random.standard_normal(nb_transits[n])
        rv_err[n] = np.sqrt(
            0.5 * np.pi * np.var(rv_mod, ddof=1) / nb_transits[n] + 0.11**2
        )
        if rv_err[n] > max_rv_err:
            rv_err[n] = np.nan

    sample_variance = 2 * nb_transits * (rv_err**2 - 0.11**2) / np.pi
    statistic = sample_variance * (nb_transits - 1)
    pval = 1 - scipy.stats.chi2(nb_transits - 1).cdf(
        statistic / rv_est_uncert**2
    ).astype(np.float32)

    data = np.empty(
        num_sims,
        dtype=[
            ("rv_est_uncert", np.float32),
            ("rv_unc_conf", np.float32),
            ("rv_pval", np.float32),
            ("dr2_rv_nb_transits", np.int32),
            ("dr2_radial_velocity_error", np.float32),
            ("sim_period", np.float32),
            ("sim_semiamp", np.float32),
            ("sim_ecc", np.float32),
            ("sim_omega", np.float32),
            ("sim_phase", np.float32),
        ],
    )
    data["rv_est_uncert"] = rv_est_uncert
    data["rv_unc_conf"] = np.ones_like(rv_est_uncert)
    data["rv_pval"] = pval
    data["dr2_rv_nb_transits"] = nb_transits
    data["dr2_radial_velocity_error"] = rv_err
    data["sim_period"] = period
    data["sim_semiamp"] = semiamp
    data["sim_ecc"] = ecc
    data["sim_omega"] = omega
    data["sim_phase"] = phase
    fitsio.write(args.output, data, clobber=True)
