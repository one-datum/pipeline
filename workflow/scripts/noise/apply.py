import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
from astropy.table import Table

jax.config.update("jax_enable_x64", True)


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, type=str)
parser.add_argument("--gp", required=True, type=str)
parser.add_argument("-o", "--output", required=True, type=str)
args = parser.parse_args()

with open(args.gp, "rb") as f:
    (params, y, gp) = pickle.load(f)


@jax.jit
def get_noise_estimate(color, mag):
    cond = gp.condition(y, (color, mag)).gp
    return cond.loc, jnp.sqrt(
        cond.variance + jnp.exp(2 * params["log_jitter"])
    )


print("Loading data...")
data = Table.read(args.input)

print("Estimating noise...")
nb_transits = data["rv_nb_transits"].value.astype(np.int32)
eps = data["radial_velocity_error"].value
sample_variance = 2 * nb_transits * (eps**2 - 0.11**2) / np.pi
statistic = sample_variance * (nb_transits - 1)
norm = np.random.default_rng(55).standard_normal(50)

step = 5000
ln_sigma = np.full(len(data), np.nan)
ln_sigma_err = np.full(len(data), np.nan)
pval = np.full(len(data), np.nan)
pval_err = np.full(len(data), np.nan)
valid = np.isfinite(data["phot_g_mean_mag"].value) & np.isfinite(
    data["bp_rp"].value
)
inds = np.arange(len(data))[valid]
for n in range(0, len(inds), step):
    i = inds[n : n + step]
    color = np.ascontiguousarray(data["bp_rp"].value[i], dtype=float)
    mag = np.ascontiguousarray(data["phot_g_mean_mag"].value[i], dtype=float)
    ln_sigma[i], ln_sigma_err[i] = get_noise_estimate(color, mag)

    arg = ln_sigma[i, None] + ln_sigma_err[i, None] * norm[None, :]
    arg = 1 - scipy.stats.chi2(nb_transits[i, None] - 1).cdf(
        statistic[i, None] * np.exp(-2 * arg)
    )
    pval[i] = np.mean(arg, axis=1)
    pval_err[i] = np.std(arg, axis=1)

data["rv_ln_uncert"] = ln_sigma
data["rv_ln_uncert_err"] = ln_sigma_err
data["rv_pval"] = pval
data["rv_pval_err"] = pval_err
data.write(args.output, overwrite=True)
