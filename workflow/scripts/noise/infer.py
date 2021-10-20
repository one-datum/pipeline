#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple

import jax.numpy as jnp
import jax.random as random
import jax.scipy as jsp
import numpy as np
import numpyro
import numpyro.distributions as dist

import tensorflow_probability.substrates.jax as tfp
from astropy.io import fits
from jax import lax, random
from jax.config import config
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.transforms import AffineTransform
from numpyro.distributions.util import (
    is_prng_key,
    promote_shapes,
    validate_sample,
)
from numpyro.infer import SVI, Trace_ELBO
from tqdm import tqdm

config.update("jax_enable_x64", True)


MIN_COLOR: float = 0.0
MAX_COLOR: float = 5.5
MIN_MAG: float = 4.5
MAX_MAG: float = 16.0


def _random_chi2(key, df, shape=(), dtype=jnp.float_):
    return 2.0 * random.gamma(key, 0.5 * df, shape=shape, dtype=dtype)


class NoncentralChi2(Distribution):
    arg_constraints = {
        "df": constraints.positive,
        "nc": constraints.positive,
    }
    support = constraints.positive
    reparametrized_params = ["df", "nc"]

    def __init__(self, df, nc, validate_args=None):
        self.df, self.nc = promote_shapes(df, nc)
        batch_shape = lax.broadcast_shapes(jnp.shape(df), jnp.shape(nc))
        super(NoncentralChi2, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def sample(self, key, sample_shape=()):
        # Ref: https://github.com/numpy/numpy/blob/89c80ba606f4346f8df2a31cfcc0e967045a68ed/numpy/random/src/distributions/distributions.c#L797-L813
        assert is_prng_key(key)
        shape = sample_shape + self.batch_shape + self.event_shape

        key1, key2, key3 = random.split(key, 3)

        i = random.poisson(key1, 0.5 * self.nc, shape=shape)
        n = random.normal(key2, shape=shape) + jnp.sqrt(self.nc)
        cond = jnp.greater(self.df, 1.0)
        chi2 = _random_chi2(
            key3,
            jnp.where(cond, self.df - 1.0, self.df + 2.0 * i),
            shape=shape,
        )
        return jnp.where(cond, chi2 + n * n, chi2)

    @validate_sample
    def log_prob(self, value):
        # Ref: https://github.com/scipy/scipy/blob/500878e88eacddc7edba93dda7d9ee5f784e50e6/scipy/stats/_distn_infrastructure.py#L597-L610
        df2 = self.df / 2.0 - 1.0
        xs, ns = jnp.sqrt(value), jnp.sqrt(self.nc)
        res = (
            jsp.special.xlogy(df2 / 2.0, value / self.nc)
            - 0.5 * (xs - ns) ** 2
        )
        corr = tfp.math.bessel_ive(df2, xs * ns) / 2.0
        return jnp.where(
            jnp.greater(corr, 0.0),
            res + jnp.log(corr),
            -jnp.inf,
        )

    @property
    def mean(self):
        return self.df + self.nc

    @property
    def variance(self):
        return 2.0 * (self.df + 2.0 * self.nc)


def setup_model(sample_variance) -> SVI:
    def model(num_transit, statistic=None):
        log_sigma = numpyro.sample("log_sigma", dist.Normal(0.0, 10.0))

        with numpyro.plate("targets", len(num_transit)):
            log_k = numpyro.sample("log_k", dist.Normal(0.0, 10.0))
            lam = num_transit * 0.5 * jnp.exp(2 * (log_k - log_sigma))
            numpyro.sample(
                "obs",
                dist.TransformedDistribution(
                    NoncentralChi2(num_transit, lam),
                    AffineTransform(loc=0.0, scale=jnp.exp(2 * log_sigma)),
                ),
                obs=statistic,
            )

    init = {
        "log_sigma": 0.5 * np.log(np.median(sample_variance)),
        "log_k": np.log(np.sqrt(sample_variance)),
    }
    guide = numpyro.infer.autoguide.AutoNormal(
        model, init_loc_fn=numpyro.infer.init_to_value(values=init)
    )
    optimizer = numpyro.optim.Adam(step_size=1e-3)
    return SVI(model, guide, optimizer, loss=numpyro.infer.Trace_ELBO())


def load_data(
    data_path: str,
    *,
    min_nb_transits: int = 3,
    color_range: Tuple[float, float] = (MIN_COLOR, MAX_COLOR),
    mag_range: Tuple[float, float] = (MIN_MAG, MAX_MAG),
) -> fits.fitsrec.FITS_rec:
    print("Loading data...")
    with fits.open(data_path) as f:
        data = f[1].data

    m = np.isfinite(data["phot_g_mean_mag"])
    m &= np.isfinite(data["bp_rp"])
    m &= np.isfinite(data["dr2_radial_velocity_error"])

    m &= data["dr2_rv_nb_transits"] > min_nb_transits

    m &= color_range[0] < data["bp_rp"]
    m &= data["bp_rp"] < color_range[1]
    m &= mag_range[0] < data["phot_g_mean_mag"]
    m &= data["phot_g_mean_mag"] < mag_range[1]

    return data[m]


def fit_data(
    data: fits.fitsrec.FITS_rec,
    *,
    num_mag_bins: int,
    num_color_bins: int,
    color_range: Tuple[float, float] = (MIN_COLOR, MAX_COLOR),
    mag_range: Tuple[float, float] = (MIN_MAG, MAX_MAG),
    num_iter: int = 5,
    targets_per_fit: int = 1000,
    num_optim: int = 5000,
    seed: int = 11239,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Parse data
    num_transit = np.ascontiguousarray(
        data["dr2_rv_nb_transits"], dtype=np.int32
    )
    eps = np.ascontiguousarray(
        data["dr2_radial_velocity_error"], dtype=np.float32
    )
    sample_variance = 2 * num_transit * (eps ** 2 - 0.11 ** 2) / np.pi
    mag = np.ascontiguousarray(data["phot_g_mean_mag"], dtype=np.float32)
    color = np.ascontiguousarray(data["bp_rp"], dtype=np.float32)

    # Setup the grid and allocate the memory
    mag_bins = np.linspace(mag_range[0], mag_range[1], num_mag_bins + 1)
    color_bins = np.linspace(
        color_range[0], color_range[1], num_color_bins + 1
    )
    mu = np.empty((len(mag_bins) - 1, len(color_bins) - 1, num_iter))
    sigma = np.empty_like(mu)
    count = np.empty((len(mag_bins) - 1, len(color_bins) - 1), dtype=np.int64)

    np.random.seed(seed)
    inds = np.arange(len(data))
    for n in tqdm(range(len(mag_bins) - 1), desc="magnitudes"):
        for m in tqdm(range(len(color_bins) - 1), desc="colors", leave=False):
            mask = mag_bins[n] <= mag
            mask &= mag <= mag_bins[n + 1]
            mask &= color_bins[m] <= color
            mask &= color <= color_bins[m + 1]

            count[n, m] = mask.sum()

            # For small amounts of data
            if count[n, m] <= targets_per_fit:
                if count[n, m] < 50:
                    mu[n, m, :] = np.nan
                    sigma[n, m, :] = np.nan
                    continue

                svi = setup_model(sample_variance[mask])
                svi_result = svi.run(
                    random.PRNGKey(seed + n + m),
                    num_optim,
                    num_transit[mask],
                    statistic=(num_transit[mask] - 1) * sample_variance[mask],
                    progress_bar=False,
                )
                params = svi_result.params
                mu[n, m, :] = params["log_sigma_auto_loc"]
                sigma[n, m, :] = params["log_sigma_auto_scale"]
                continue

            for k in tqdm(range(num_iter), desc="iterations", leave=False):
                masked_inds = np.random.choice(
                    inds[mask],
                    size=targets_per_fit,
                    replace=mask.sum() <= targets_per_fit,
                )

                svi = setup_model(sample_variance[masked_inds])
                svi_result = svi.run(
                    random.PRNGKey(seed + n + m + k),
                    num_optim,
                    num_transit[masked_inds],
                    statistic=(num_transit[masked_inds] - 1)
                    * sample_variance[masked_inds],
                    progress_bar=False,
                )
                params = svi_result.params
                mu[n, m, k] = params["log_sigma_auto_loc"]
                sigma[n, m, k] = params["log_sigma_auto_scale"]

    return mu, sigma, count


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
        data_path=args.input,
        min_nb_transits=config["min_nb_transits"],
        color_range=(
            config["min_color"],
            config["max_color"],
        ),
        mag_range=(
            config["min_mag"],
            config["max_mag"],
        ),
    )

    mu, sigma, count = fit_data(
        data,
        num_mag_bins=config["num_mag"],
        num_color_bins=config["num_color"],
        color_range=(
            config["min_color"],
            config["max_color"],
        ),
        mag_range=(
            config["min_mag"],
            config["max_mag"],
        ),
        num_iter=config["num_iter"],
        targets_per_fit=config["targets_per_fit"],
        num_optim=config["num_optim"],
        seed=config["seed"],
    )

    # Save the results
    hdr = fits.Header()
    hdr["min_tra"] = config["min_nb_transits"]
    hdr["min_col"] = config["min_color"]
    hdr["max_col"] = config["max_color"]
    hdr["num_col"] = config["num_color"]
    hdr["min_mag"] = config["min_mag"]
    hdr["max_mag"] = config["max_mag"]
    hdr["num_mag"] = config["num_mag"]
    hdr["num_itr"] = config["num_iter"]
    hdr["num_per"] = config["targets_per_fit"]
    hdr["num_opt"] = config["num_optim"]
    hdr["seed"] = config["seed"]
    fits.HDUList(
        [
            fits.PrimaryHDU(header=hdr),
            fits.ImageHDU(mu),
            fits.ImageHDU(sigma),
            fits.ImageHDU(count),
        ]
    ).writeto(args.output, overwrite=True)
