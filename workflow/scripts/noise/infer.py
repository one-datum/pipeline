#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from astropy.io import fits
from jax import random
from jax.config import config as jax_config
from jax.interpreters import xla
from numpyro.infer import SVI, Trace_ELBO
from tqdm import tqdm

jax_config.update("jax_enable_x64", True)


MIN_COLOR: float = 0.0
MAX_COLOR: float = 5.5
MIN_MAG: float = 4.5
MAX_MAG: float = 16.0


def setup_model() -> SVI:
    def model(num_transit, log_plx, stat=None):
        m = numpyro.sample("m", dist.Normal(0.0, 10.0))
        b = numpyro.sample("b", dist.Normal(0.0, 10.0))
        log_sigma0 = numpyro.deterministic("log_sigma0", m * log_plx + b)
        log_dsigma = numpyro.sample(
            "log_dsigma",
            dist.Normal(0.0, 10.0),
            sample_shape=(len(num_transit),),
        )
        sigma2 = jnp.exp(2 * log_sigma0) + jnp.exp(2 * log_dsigma)
        numpyro.sample(
            "obs", dist.Gamma(0.5 * (num_transit - 1), 0.5 / sigma2), obs=stat
        )

    def make_mean_field(name, init):
        mu = numpyro.param(f"mu_{name}", init)
        sigma = numpyro.param(
            f"sigma_{name}",
            1.0 + 0.0 * init,
            constraint=dist.constraints.positive,
        )
        return numpyro.sample(name, dist.Normal(mu, sigma))

    def guide(num_transit, log_plx, stat=None):
        numpyro.param("m", 0.0)
        numpyro.param("b", 0.0)
        make_mean_field(
            "log_dsigma",
            0.5
            * jnp.log(
                jnp.ones_like(log_plx)
                if stat is None
                else (stat / (num_transit - 1))
            ),
        )

    optimizer = numpyro.optim.Adam(step_size=2e-1)
    return SVI(model, guide, optimizer, loss=Trace_ELBO())


def load_data(
    data_path: str,
    *,
    min_nb_transits: int = 3,
) -> fits.fitsrec.FITS_rec:
    print("Loading data...")
    with fits.open(data_path) as f:
        data = f[1].data

    m = np.isfinite(data["phot_g_mean_mag"])
    m &= np.isfinite(data["bp_rp"])
    m &= np.isfinite(data["dr2_radial_velocity_error"])
    m &= np.isfinite(data["parallax"])
    m &= data["parallax"] > 0

    m &= data["dr2_rv_nb_transits"] > min_nb_transits

    # Remove colors and manitudes with 1e+20 entries
    m &= data["phot_g_mean_mag"] < 100
    m &= data["bp_rp"] < 100

    return data[m]


@partial(jax.jit, static_argnums=(0, 1))
def train(svi, num_optim, key, num_transit, log_plx, stat):
    params, _ = svi.run(
        key,
        num_optim,
        num_transit,
        log_plx,
        stat,
        progress_bar=False,
    )
    return params["b"], params["m"]


def fit_data(
    data: fits.fitsrec.FITS_rec,
    *,
    num_mag_bins: int,
    num_color_bins: int,
    num_optim: int = 1000,
    seed: int = 11239,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    # Parse data
    num_transit = np.ascontiguousarray(
        data["dr2_rv_nb_transits"], dtype=np.int32
    )
    eps = np.ascontiguousarray(
        data["dr2_radial_velocity_error"], dtype=np.float32
    )
    sample_variance = 2 * num_transit * (eps ** 2 - 0.11 ** 2) / np.pi
    stat = sample_variance * (num_transit - 1)
    log_plx = np.log(data["parallax"])
    mag = np.ascontiguousarray(data["phot_g_mean_mag"], dtype=np.float32)
    color = np.ascontiguousarray(data["bp_rp"], dtype=np.float32)

    # Setup the JAX model
    svi = setup_model()

    # Setup the grid and allocate the memory
    color_bins = np.percentile(
        data["bp_rp"], np.linspace(0, 100, num_color_bins + 1)
    )
    mag_bins = np.percentile(
        data["phot_g_mean_mag"], np.linspace(0, 100, num_mag_bins + 1)
    )
    intercept = np.full((len(mag_bins) - 1, len(color_bins) - 1), np.nan)
    slope = np.full((len(mag_bins) - 1, len(color_bins) - 1), np.nan)
    count = np.zeros((len(mag_bins) - 1, len(color_bins) - 1), dtype=np.int64)

    for n in tqdm(range(len(mag_bins) - 1), desc="magnitudes"):
        for m in tqdm(range(len(color_bins) - 1), desc="colors", leave=False):
            mask = mag_bins[n] <= mag
            mask &= mag <= mag_bins[n + 1]
            mask &= color_bins[m] <= color
            mask &= color <= color_bins[m + 1]

            count[n, m] = mask.sum()
            if count[n, m] <= 1:
                continue

            intercept[n, m], slope[n, m] = train(
                svi,
                num_optim,
                random.PRNGKey(seed + n + m),
                num_transit[mask],
                log_plx[mask],
                stat[mask],
            )

        # Try not to have JAX memory footprint grow forever
        xla._xla_callable.cache_clear()

    return intercept, slope, count, (mag_bins, color_bins)


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

    data = load_data(
        data_path=args.input, min_nb_transits=config["min_nb_transits"]
    )

    intercept, slope, count, bins = fit_data(
        data,
        num_mag_bins=config["num_mag"],
        num_color_bins=config["num_color"],
        num_optim=config["num_optim"],
        seed=config["seed"],
    )

    # Save the results
    hdr = fits.Header()
    hdr["min_tra"] = config["min_nb_transits"]
    hdr["num_col"] = config["num_color"]
    hdr["num_mag"] = config["num_mag"]
    hdr["num_opt"] = config["num_optim"]
    hdr["seed"] = config["seed"]
    fits.HDUList(
        [
            fits.PrimaryHDU(header=hdr),
            fits.ImageHDU(intercept),
            fits.ImageHDU(slope),
            fits.ImageHDU(count),
            fits.ImageHDU(bins[0], name="mag bins"),
            fits.ImageHDU(bins[1], name="color bins"),
        ]
    ).writeto(args.output, overwrite=True)
