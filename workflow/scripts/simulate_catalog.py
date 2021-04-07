#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fitsio
import kepler
import numpy as np
import yaml

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-c", "--config", required=True, type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    print("Loading data...")
    data = fitsio.read(
        args.input, columns=["dr2_rv_nb_transits", "dr2_radial_velocity_error"]
    )
    max_rv_err = data["dr2_radial_velocity_error"].max()

    print("Simulating model...")
    random = np.random.default_rng(config["seed"])
    num_sims = int(config["num_sims"])

    nb_transits = data["dr2_rv_nb_transits"][
        random.integers(0, len(data), num_sims)
    ]

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
    ecc = random.uniform(config["ecc"]["min"], config["ecc"]["max"], num_sims)
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
            0.5 * np.pi * np.var(rv_mod, ddof=1) / nb_transits[n] + 0.11 ** 2
        )
        if rv_err[n] > max_rv_err:
            rv_err[n] = np.nan

    data = np.empty(
        num_sims,
        dtype=[
            ("rv_est_uncert", np.float32),
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
    data["dr2_rv_nb_transits"] = nb_transits
    data["dr2_radial_velocity_error"] = rv_err
    data["sim_period"] = period
    data["sim_semiamp"] = semiamp
    data["sim_ecc"] = ecc
    data["sim_omega"] = omega
    data["sim_phase"] = phase
    fitsio.write(args.output, data, clobber=True)
