#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple

import numpy as np
import scipy.stats
from tqdm import tqdm
from astropy.io import fits

# import jax.numpy as jnp
# import numpyro
# import numpyro.distributions as dist
# from jax import random
# from jax.config import config as jax_config
# from numpyro.distributions.transforms import AffineTransform
# from numpyro.infer import SVI
# from numpyro_ext.distributions import NoncentralChi2

# jax_config.update("jax_enable_x64", True)


MIN_COLOR: float = 0.0
MAX_COLOR: float = 5.5
MIN_MAG: float = 4.5
MAX_MAG: float = 13.0


# def setup_model(sample_variance) -> SVI:
#     def model(num_transit, statistic=None):
#         log_sigma = numpyro.sample("log_sigma", dist.Normal(0.0, 10.0))

#         with numpyro.plate("targets", len(num_transit)):
#             log_k = numpyro.sample("log_k", dist.Normal(0.0, 10.0))
#             lam = num_transit * 0.5 * jnp.exp(2 * (log_k - log_sigma))
#             numpyro.sample(
#                 "obs",
#                 dist.TransformedDistribution(
#                     NoncentralChi2(num_transit - 1, lam),
#                     AffineTransform(loc=0.0, scale=jnp.exp(2 * log_sigma)),
#                 ),
#                 obs=statistic,
#             )

#     init = {
#         "log_sigma": 0.5 * np.log(np.median(sample_variance)),
#         "log_k": np.log(np.sqrt(sample_variance)),
#     }
#     guide = numpyro.infer.autoguide.AutoNormal(
#         model, init_loc_fn=numpyro.infer.init_to_value(values=init)
#     )
#     optimizer = numpyro.optim.Adam(step_size=1e-3)
#     return SVI(model, guide, optimizer, loss=numpyro.infer.Trace_ELBO())


def esimate_sigma(
    num_transit, sample_variance, pval_threshold=0.001, maxiter=100
):
    for i in range(maxiter):
        stat = sample_variance * (num_transit - 1)
        var_est = np.median(sample_variance)
        pval = 1 - scipy.stats.chi2(num_transit - 1).cdf(stat / var_est)
        m = pval > pval_threshold
        if np.all(m):
            break
        num_transit = num_transit[m]
        sample_variance = sample_variance[m]
    return np.sqrt(var_est)


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
    m &= np.isfinite(data["radial_velocity_error"])

    m &= data["rv_nb_transits"] > min_nb_transits

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
    num_transit = np.ascontiguousarray(data["rv_nb_transits"], dtype=np.int32)
    eps = np.ascontiguousarray(data["radial_velocity_error"], dtype=np.float32)
    sample_variance = 2 * num_transit * (eps**2 - 0.11**2) / np.pi
    mag = np.ascontiguousarray(data["phot_g_mean_mag"], dtype=np.float32)
    color = np.ascontiguousarray(data["bp_rp"], dtype=np.float32)

    # Setup the grid and allocate the memory
    mag_bins = np.linspace(mag_range[0], mag_range[1], num_mag_bins + 1)
    color_bins = np.linspace(
        color_range[0], color_range[1], num_color_bins + 1
    )
    mu = np.empty((len(mag_bins) - 1, len(color_bins) - 1, num_iter))
    # sigma = np.empty_like(mu)
    count = np.empty((len(mag_bins) - 1, len(color_bins) - 1), dtype=np.int64)

    random = np.random.default_rng(seed)
    inds = np.arange(len(data))
    for n in tqdm(range(len(mag_bins) - 1), desc="magnitudes"):
        for m in tqdm(range(len(color_bins) - 1), desc="colors", leave=False):
            mask = mag_bins[n] <= mag
            mask &= mag <= mag_bins[n + 1]
            mask &= color_bins[m] <= color
            mask &= color <= color_bins[m + 1]

            count[n, m] = mask.sum()

            if count[n, m] < 50:
                mu[n, m, :] = np.nan
                # sigma[n, m, :] = np.nan
                continue

            for k in range(num_iter):
                inds = random.choice(mask.sum(), size=targets_per_fit)
                mu[n, m, k] = esimate_sigma(
                    num_transit[mask][inds], sample_variance[mask][inds]
                )

            # # For small amounts of data
            # if count[n, m] <= targets_per_fit:
            #     if count[n, m] < 50:
            #         mu[n, m, :] = np.nan
            #         sigma[n, m, :] = np.nan
            #         continue

            #     svi = setup_model(sample_variance[mask])
            #     svi_result = svi.run(
            #         random.PRNGKey(seed + n + m),
            #         num_optim,
            #         num_transit[mask],
            #         statistic=(num_transit[mask] - 1) * sample_variance[mask],
            #         progress_bar=False,
            #     )
            #     params = svi_result.params
            #     mu[n, m, :] = params["log_sigma_auto_loc"]
            #     sigma[n, m, :] = params["log_sigma_auto_scale"]
            #     continue

            # for k in tqdm(range(num_iter), desc="iterations", leave=False):
            #     masked_inds = np.random.choice(
            #         inds[mask],
            #         size=targets_per_fit,
            #         replace=mask.sum() <= targets_per_fit,
            #     )

            #     svi = setup_model(sample_variance[masked_inds])
            #     svi_result = svi.run(
            #         random.PRNGKey(seed + n + m + k),
            #         num_optim,
            #         num_transit[masked_inds],
            #         statistic=(num_transit[masked_inds] - 1)
            #         * sample_variance[masked_inds],
            #         progress_bar=False,
            #     )
            #     params = svi_result.params
            #     mu[n, m, k] = params["log_sigma_auto_loc"]
            #     sigma[n, m, k] = params["log_sigma_auto_scale"]

    return mu, count
    # return mu, sigma, count


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--mag-bin", required=True, type=int)
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-c", "--config", required=True, type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    mag_bins = np.linspace(
        config["min_mag"], config["max_mag"], config["num_mag"] + 1
    )
    min_mag = mag_bins[args.mag_bin]
    max_mag = mag_bins[args.mag_bin + 1]

    data = load_data(
        data_path=args.input,
        min_nb_transits=config["min_nb_transits"],
        color_range=(
            config["min_color"],
            config["max_color"],
        ),
        mag_range=(min_mag, max_mag),
    )

    # mu, sigma, count = fit_data(
    mu, count = fit_data(
        data,
        num_mag_bins=1,
        num_color_bins=config["num_color"],
        color_range=(
            config["min_color"],
            config["max_color"],
        ),
        mag_range=(min_mag, max_mag),
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
            fits.ImageHDU(count),
        ]
    ).writeto(args.output, overwrite=True)
