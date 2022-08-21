import argparse

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
import tqdm
from astropy.table import Table


def kepler(M, ecc):
    # Wrap into the right range
    M = M % (2 * jnp.pi)

    # We can restrict to the range [0, pi)
    high = M > jnp.pi
    M = jnp.where(high, 2 * jnp.pi - M, M)

    # Solve
    ome = 1 - ecc
    E = starter(M, ecc, ome)
    E = refine(M, ecc, ome, E)

    # Re-wrap back into the full range
    E = jnp.where(high, 2 * jnp.pi - E, E)

    # Convert to true anomaly; tan(0.5 * f)
    tan_half_f = jnp.sqrt((1 + ecc) / (1 - ecc)) * jnp.tan(0.5 * E)
    tan2_half_f = jnp.square(tan_half_f)

    # Then we compute sin(f) and cos(f) using:
    #  sin(f) = 2*tan(0.5*f)/(1 + tan(0.5*f)^2), and
    #  cos(f) = (1 - tan(0.5*f)^2)/(1 + tan(0.5*f)^2)
    denom = 1 / (1 + tan2_half_f)
    sinf = 2 * tan_half_f * denom
    cosf = (1 - tan2_half_f) * denom

    return sinf, cosf


def starter(M, ecc, ome):
    M2 = jnp.square(M)
    alpha = 3 * jnp.pi / (jnp.pi - 6 / jnp.pi)
    alpha += 1.6 / (jnp.pi - 6 / jnp.pi) * (jnp.pi - M) / (1 + ecc)
    d = 3 * ome + alpha * ecc
    alphad = alpha * d
    r = (3 * alphad * (d - ome) + M2) * M
    q = 2 * alphad * ome - M2
    q2 = jnp.square(q)
    w = (jnp.abs(r) + jnp.sqrt(q2 * q + r * r)) ** (2.0 / 3)
    return (2 * r * w / (jnp.square(w) + w * q + q2) + M) / d


def refine(M, ecc, ome, E):
    sE = E - jnp.sin(E)
    cE = 1 - jnp.cos(E)

    f_0 = ecc * sE + E * ome - M
    f_1 = ecc * cE + ome
    f_2 = ecc * (E - sE)
    f_3 = 1 - f_1
    d_3 = -f_0 / (f_1 - 0.5 * f_0 * f_2 / f_1)
    d_4 = -f_0 / (f_1 + 0.5 * d_3 * f_2 + (d_3 * d_3) * f_3 / 6)
    d_42 = d_4 * d_4
    dE = -f_0 / (
        f_1 + 0.5 * d_4 * f_2 + d_4 * d_4 * f_3 / 6 - d_42 * d_4 * f_2 / 24
    )

    return E + dE


def ncx2_logprob(df, nc, value):
    # Ref: https://github.com/scipy/scipy/blob/
    # 500878e88eacddc7edba93dda7d9ee5f784e50e6/scipy/
    # stats/_distn_infrastructure.py#L597-L610
    df2 = df / 2.0 - 1.0
    xs, ns = jnp.sqrt(value), jnp.sqrt(nc)
    res = jsp.special.xlogy(df2 / 2.0, value / nc) - 0.5 * (xs - ns) ** 2
    corr = tfp.math.bessel_ive(df2, xs * ns) / 2.0
    return jnp.where(
        jnp.greater(corr, 0.0),
        res + jnp.log(corr),
        -jnp.inf,
    )


@jax.jit
def get_logpdf(t, ln_sigma, ln_sigma_err, nb_transits, statistic):
    # Compute the Keplerian model
    M = factor * t + phase
    sinf, cosf = kepler(M, ecc)
    rv_mod = kcosw * (ecc + cosf) - ksinw * sinf
    lam = jnp.sum((rv_mod - jnp.mean(rv_mod, axis=0)[None, :]) ** 2, axis=0)

    ivar = jnp.exp(-2 * (ln_sigma + ln_sigma_err * norm))
    return ncx2_logprob(
        df=nb_transits - 1, nc=lam * ivar, value=statistic * ivar
    )


@jax.jit
def fit_one(t, ln_sigma, ln_sigma_err, nb_transits, statistic):
    log_weight = get_logpdf(t, ln_sigma, ln_sigma_err, nb_transits, statistic)
    weights = jnp.exp(log_weight - log_weight.max())
    cdf = jnp.cumsum(weights)
    return jnp.asarray(K)[
        jnp.clip(
            jnp.searchsorted(cdf, QUANTILES * cdf[-1]) - 1, 0, len(cdf) - 1
        )
    ]


fit_batch = jax.jit(jax.vmap(fit_one))

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, type=str)
parser.add_argument("-o", "--output", required=True, type=str)
args = parser.parse_args()

data = Table.read(args.input)
pval = np.ascontiguousarray(data["rv_pval"].value, dtype=np.float32)
ln_sigma = np.ascontiguousarray(data["rv_ln_uncert"].value, dtype=np.float32)
ln_sigma_err = np.ascontiguousarray(
    data["rv_ln_uncert_err"].value, dtype=np.float32
)
nb_transits = np.ascontiguousarray(
    data["rv_nb_transits"].value, dtype=np.int32
)
time_duration = np.ascontiguousarray(
    data["rv_time_duration"].value, dtype=np.float32
)
eps = np.ascontiguousarray(
    data["radial_velocity_error"].value, dtype=np.float32
)
sample_variance = 2 * nb_transits * (eps**2 - 0.11**2) / np.pi
statistic = sample_variance * (nb_transits - 1)
valid = np.isfinite(ln_sigma) & np.isfinite(eps) & np.isfinite(time_duration)


num_samp = 5_000
QUANTILES = np.array([0.05, 0.16, 0.5, 0.84, 0.95])

random = np.random.default_rng(0)

# Sample many parameters from the prior
log_semiamp = np.sort(random.uniform(np.log(0.05), np.log(500.0), num_samp))
K = np.exp(log_semiamp)
log_period = random.uniform(np.log(1.0), np.log(800.0), num_samp)[None, :]
phase = random.uniform(-np.pi, np.pi, num_samp)[None, :]
# ecc = random.beta(0.867, 3.03, num_samp)
ecc = random.uniform(0.0, 0.9, num_samp)[None, :]
omega = random.uniform(-np.pi, np.pi, num_samp)[None, :]
kcosw = K[None, :] * np.cos(omega)
ksinw = K[None, :] * np.sin(omega)
factor = 2 * np.pi * np.exp(-log_period)
norm = random.standard_normal(num_samp)

step = 10
all_inds = np.arange(len(data))
results = np.full((len(data), len(QUANTILES)), np.nan)
for target_num in np.unique(nb_transits):
    print(f"rv_nb_transits = {target_num}")
    t_frac = random.uniform(0, 1, (target_num, num_samp))
    inds = all_inds[(nb_transits == target_num) & valid & (pval < 0.01)]
    for n in tqdm.trange(0, len(inds), step):
        i = inds[n : n + step]
        results[i] = fit_batch(
            time_duration[i, None, None] * t_frac[None],
            ln_sigma[i],
            ln_sigma_err[i],
            nb_transits[i],
            statistic[i],
        )

for n, q in enumerate(QUANTILES):
    data[f"rv_semiamp_p{(100 * q):.0f}"] = results[:, n]
data.write(args.output, overwrite=True)


# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# from functools import partial
# from multiprocessing import Pool
# from multiprocessing.managers import SharedMemoryManager
# from multiprocessing.shared_memory import SharedMemory
# from typing import Tuple

# import fitsio
# import kepler
# import numpy as np
# import scipy.stats
# from numpy.lib.recfunctions import append_fields

# QUANTILES = np.array([0.05, 0.16, 0.5, 0.84, 0.95])


# def precompute_model(
#     max_nb_transits: int, *, num_samp: int = 50000, seed: int = 384820
# ) -> Tuple[np.ndarray, np.ndarray]:
#     random = np.random.default_rng(seed)

#     # Simulate transit times by sampling from the baseline
#     t = random.uniform(0, 668, (max_nb_transits, num_samp))

#     # Sample many parameters from the prior
#     log_period = random.uniform(np.log(1.0), np.log(800.0), num_samp)
#     phase = random.uniform(-np.pi, np.pi, num_samp)
#     log_semiamp = np.sort(
#         random.uniform(np.log(0.1), np.log(1000.0), num_samp)
#     )
#     # ecc = random.beta(0.867, 3.03, num_samp)
#     ecc = random.uniform(0.0, 0.9, num_samp)
#     omega = random.uniform(-np.pi, np.pi, num_samp)

#     # Compute the Keplerian model
#     cosw = np.cos(omega)
#     sinw = np.sin(omega)
#     M = 2 * np.pi * t * np.exp(-log_period)[None, :] + phase[None, :]
#     _, cosf, sinf = kepler.kepler(M, ecc[None, :] + np.zeros_like(M))
#     rv_mod = np.exp(log_semiamp[None, :]) * (
#         cosw[None, :] * (ecc[None, :] + cosf) - sinw[None, :] * sinf
#     )

#     lam = np.zeros((max_nb_transits + 1, num_samp), dtype=np.float32)
#     for n in range(2, max_nb_transits + 1):
#         m = rv_mod[: n + 1]
#         lam[n] = np.sum((m - np.mean(m, axis=0)[None, :]) ** 2, axis=0)

#     return log_semiamp.astype(np.float32), lam


# def _inner_process(
#     rate_param: np.ndarray,
#     ln_sigma: np.ndarray,
#     nb_transits: np.ndarray,
#     statistic: np.ndarray,
# ):
#     # The Keplerian model
#     ivar = np.exp(-2 * ln_sigma)
#     target_lam = rate_param[nb_transits - 1] * ivar[:, None]
#     ncx2 = scipy.stats.ncx2(df=nb_transits[:, None] - 1, nc=target_lam)

#     s2 = np.multiply(statistic, ivar, out=ivar)
#     log_weight = ncx2.logpdf(s2[:, None])
#     weights = np.exp(
#         log_weight - log_weight.max(axis=1)[:, None], out=target_lam
#     )

#     # Compute the quantiles assuming that ln_semiamp is sorted
#     cdf = np.cumsum(weights, axis=1, out=weights)
#     return np.array(
#         [
#             np.clip(np.searchsorted(c, QUANTILES * c[-1]) - 1, 0, len(c) - 1)
#             for c in cdf
#         ]
#     )


# def process(
#     rate_param: np.ndarray,
#     args: Tuple[np.ndarray, np.ndarray, np.ndarray],
# ) -> np.ndarray:
#     return _inner_process(rate_param, *args)


# def process_shared(
#     name: str,
#     shape: Tuple[int],
#     dtype: np.dtype,
#     args: Tuple[np.ndarray, np.ndarray, np.ndarray],
# ) -> np.ndarray:
#     rate_buf = SharedMemory(name)
#     rate_param = np.ndarray(shape=shape, dtype=dtype, buffer=rate_buf.buf)
#     return _inner_process(rate_param, *args)


# if __name__ == "__main__":
#     import argparse
#     import time
#     import tracemalloc

#     parser = argparse.ArgumentParser()
#     parser.add_argument("-i", "--input", required=True, type=str)
#     parser.add_argument("-o", "--output", required=True, type=str)
#     parser.add_argument("-s", "--block-size", default=10, type=int)
#     args = parser.parse_args()

#     block_size = int(args.block_size)

#     print("Loading data...")
#     data = fitsio.read(args.input)

#     # Compute the ingredients for probabilistic model
#     ln_sigma = np.log(data["rv_est_uncert"])
#     nb_transits = data["dr2_rv_nb_transits"].astype(np.int32)
#     eps = data["dr2_radial_velocity_error"]
#     sample_variance = 2 * nb_transits * (eps**2 - 0.11**2) / np.pi
#     statistic = (sample_variance * (nb_transits - 1)).astype(np.float32)
#     pval = data["rv_pval"]
#     valid = (
#         (nb_transits >= 3)
#         & (pval < 0.01)
#         & np.isfinite(ln_sigma)
#         & np.isfinite(eps)
#     )
#     ln_sigma = ln_sigma[valid]
#     nb_transits = nb_transits[valid]
#     statistic = statistic[valid]
#     max_nb_transits = data["dr2_rv_nb_transits"].max()
#     num_rows = len(statistic)

#     # Simulate the RV error model
#     print("Simulating model...")
#     ln_semiamp, rate_param = precompute_model(max_nb_transits)

#     print("Processing shared...")
#     tracemalloc.start()
#     start_time = time.time()
#     with SharedMemoryManager() as smm:
#         shared_rate_param = smm.SharedMemory(rate_param.nbytes)  # type: ignore
#         rate_array = np.ndarray(
#             shape=rate_param.shape,
#             dtype=rate_param.dtype,
#             buffer=shared_rate_param.buf,
#         )
#         rate_array[:] = rate_param

#         func = partial(
#             process_shared,
#             shared_rate_param.name,
#             rate_param.shape,
#             rate_param.dtype,
#         )
#         with Pool() as pool:
#             results = list(
#                 pool.map(
#                     func,
#                     [
#                         (
#                             ln_sigma[n : n + block_size],
#                             nb_transits[n : n + block_size],
#                             statistic[n : n + block_size],
#                         )
#                         for n in range(0, num_rows, block_size)
#                     ],
#                 )
#             )
#     current, peak = tracemalloc.get_traced_memory()
#     print(f"Current memory usage {current/1e6}MB; Peak: {peak/1e6}MB")
#     print(f"Time elapsed: {time.time()-start_time:.2f}s")
#     tracemalloc.stop()

#     # Save the results
#     inds = np.concatenate(results, axis=0)
#     result = np.empty((len(data), len(QUANTILES)), dtype=np.float32)
#     result[:] = np.nan
#     result[valid] = ln_semiamp[inds]
#     data = append_fields(
#         data,
#         ["rv_variance"] + [f"rv_semiamp_p{(100 * q):.0f}" for q in QUANTILES],
#         [sample_variance]
#         + [np.exp(result[:, q]) for q in range(len(QUANTILES))],
#     )
#     fitsio.write(args.output, data, clobber=True)
