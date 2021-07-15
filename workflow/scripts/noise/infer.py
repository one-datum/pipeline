#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
from multiprocessing import Pool
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpyro
import numpyro.distributions as dist
import tensorflow_probability.substrates.jax as tfp
from astropy.io import fits
from jax import lax, random
from jax.config import config as jax_config
from jax.interpreters import xla
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import (
    is_prng_key,
    promote_shapes,
    validate_sample,
)
from numpyro.infer import SVI, Trace_ELBO
from tqdm import tqdm

jax_config.update("jax_enable_x64", True)


def _random_chi2(key, df, shape=(), dtype=jnp.float_):
    """Generate a chi^2 distributed random number"""
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
        # Ref: https://github.com/numpy/numpy/blob/89c80ba606f4346f8df2a31cfcc0e967045a68ed/numpy/random/src/distributions/distributions.c#L797-L813  # noqa
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
        # Ref: https://github.com/scipy/scipy/blob/500878e88eacddc7edba93dda7d9ee5f784e50e6/scipy/stats/_distn_infrastructure.py#L597-L610  # noqa
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


def setup_model() -> SVI:
    def model(num_transit, stat=None):
        # The baseline error is T-distributed
        df = numpyro.sample("df", dist.Uniform(0.0, 10.0))
        loc = numpyro.sample("loc", dist.Normal(0.0, 10.0))
        log_scale = numpyro.sample("log_scale", dist.Normal(0.0, 10.0))
        log_sigma = numpyro.sample(
            "log_sigma",
            dist.StudentT(df, loc=loc, scale=jnp.exp(log_scale)),
            sample_shape=num_transit.shape,
        )

        # Excess variance
        log_extra = numpyro.sample(
            "log_extra", dist.Normal(0.0, 10.0), sample_shape=num_transit.shape
        )
        nc = jnp.exp(log_extra - 2 * log_sigma) / (num_transit + 1)

        # Then the observed statistic is a scaled noncentral chi^2
        numpyro.sample(
            "obs",
            dist.TransformedDistribution(
                NoncentralChi2(num_transit, nc),
                dist.transforms.AffineTransform(
                    loc=0.0, scale=jnp.exp(2 * log_sigma)
                ),
            ),
            obs=stat,
        )

    def make_mean_field(name, init, **kwargs):
        mu = numpyro.param(f"mu_{name}", init, **kwargs)
        sigma = numpyro.param(
            f"sigma_{name}",
            1.0 + 0.0 * init,
            constraint=dist.constraints.interval(0.0, 2.0),
        )
        return numpyro.sample(name, dist.Normal(mu, sigma))

    def guide(num_transit, stat=None):
        numpyro.param(
            "df", 4.0, constraint=dist.constraints.interval(0.0, 10.0)
        )
        numpyro.param("loc", 0.0)
        numpyro.param("log_scale", np.log(0.2))
        if stat is None:
            make_mean_field("log_sigma", np.zeros(num_transit.shape))
            make_mean_field("log_extra", np.zeros(num_transit.shape))
        else:
            make_mean_field("log_sigma", 0.5 * jnp.log(stat))
            make_mean_field("log_extra", jnp.log(stat))

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


def fit_one(num_optim, args):
    seed, num_transit, stat = args

    svi = setup_model()
    train = jax.jit(partial(svi.run, progress_bar=False), static_argnums=1)
    results = train(random.PRNGKey(seed), num_optim, num_transit, stat)
    params = results.params

    # Try not to have JAX memory footprint grow forever; prolly will anyway
    xla._xla_callable.cache_clear()

    return (
        np.log(params["df"]),
        float(params["loc"]),
        float(params["log_scale"]),
    )


def fit_data(
    data_path: str,
    *,
    min_nb_transits: int = 3,
    num_mag_bins: int,
    num_color_bins: int,
    num_optim: int = 1000,
    seed: int = 11239,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Tuple[np.ndarray, np.ndarray],
]:
    data = load_data(data_path=data_path, min_nb_transits=min_nb_transits)

    # Parse data
    num_transit = np.ascontiguousarray(
        data["dr2_rv_nb_transits"], dtype=np.int32
    )
    eps = np.ascontiguousarray(
        data["dr2_radial_velocity_error"], dtype=np.float32
    )
    sample_variance = 2 * num_transit * (eps ** 2 - 0.11 ** 2) / np.pi
    stat = sample_variance * (num_transit - 1)
    mag = np.ascontiguousarray(data["phot_g_mean_mag"], dtype=np.float32)
    color = np.ascontiguousarray(data["bp_rp"], dtype=np.float32)

    # Setup the grid and allocate the memory
    color_bins = np.percentile(
        data["bp_rp"], np.linspace(0, 100, num_color_bins + 1)
    )
    mag_bins = np.percentile(
        data["phot_g_mean_mag"], np.linspace(0, 100, num_mag_bins + 1)
    )

    log_df = np.full((len(mag_bins) - 1, len(color_bins) - 1), np.nan)
    loc = np.full((len(mag_bins) - 1, len(color_bins) - 1), np.nan)
    log_scale = np.full((len(mag_bins) - 1, len(color_bins) - 1), np.nan)
    eff_mag = np.full((len(mag_bins) - 1, len(color_bins) - 1), np.nan)
    eff_color = np.full((len(mag_bins) - 1, len(color_bins) - 1), np.nan)
    count = np.zeros((len(mag_bins) - 1, len(color_bins) - 1), dtype=np.int64)

    coords = []
    args = []

    for n in range(len(mag_bins) - 1):
        for m in range(len(color_bins) - 1):
            mask = mag_bins[n] <= mag
            mask &= mag <= mag_bins[n + 1]
            mask &= color_bins[m] <= color
            mask &= color <= color_bins[m + 1]

            # Skip if there aren't enough targets in the bin
            count[n, m] = mask.sum()
            if count[n, m] <= 1:
                continue

            # Save the effective coordaintes of the bin
            eff_mag[n, m] = np.median(data["phot_g_mean_mag"][mask])
            eff_color[n, m] = np.median(data["bp_rp"][mask])

            coords.append((n, m))
            args.append((seed + n + m, num_transit[mask], stat[mask]))

    with Pool() as pool:
        for (n, m), results in tqdm(
            zip(coords, pool.map(partial(fit_one, num_optim), args)),
            total=len(args),
        ):
            log_df[n, m] = results[0]
            loc[n, m] = results[1]
            log_scale[n, m] = results[2]

    return (
        log_df,
        loc,
        log_scale,
        count,
        eff_mag,
        eff_color,
        (mag_bins, color_bins),
    )


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

    log_df, loc, log_scale, count, mag, color, bins = fit_data(
        args.input,
        min_nb_transits=config["min_nb_transits"],
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
            fits.ImageHDU(log_df, name="log df"),
            fits.ImageHDU(loc, name="loc"),
            fits.ImageHDU(log_scale, name="log scale"),
            fits.ImageHDU(count, name="count"),
            fits.ImageHDU(mag, name="eff mag"),
            fits.ImageHDU(color, name="eff color"),
            fits.ImageHDU(bins[0], name="mag bins"),
            fits.ImageHDU(bins[1], name="color bins"),
        ]
    ).writeto(args.output, overwrite=True)
