# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %matplotlib inline

# +
import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

from jax import random
from jax.config import config
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO

import kepler

config.update("jax_enable_x64", True)


# -


def generate_dataset(
    N=1000,
    true_sigma=0.5,
    true_sigma_err=0.01,
    max_nb_transits=15,
    binary_fraction=0.3,
):
    rv_nb_transits = np.random.randint(3, max_nb_transits, N)
    times = np.random.uniform(0, 668, (max_nb_transits, N))
    sigma = np.exp(np.log(true_sigma) + true_sigma_err * np.random.randn(N))

    semi_amp = np.exp(np.random.uniform(np.log(0.1), np.log(100.0), N))
    semi_amp[np.random.rand(N) > binary_fraction] = 0.0

    log_period = np.random.uniform(np.log(1.0), np.log(1000.0), N)
    phase = np.random.uniform(-np.pi, np.pi, N)
    ecc = np.random.uniform(0, 0.8, N)
    omega = np.random.uniform(-np.pi, np.pi, N)

    mean_anom = (
        2 * np.pi * times * np.exp(-log_period)[None, :] + phase[None, :]
    )
    cosw = np.cos(omega)
    sinw = np.sin(omega)
    _, cosf, sinf = kepler.kepler(
        mean_anom, ecc[None, :] + np.zeros_like(mean_anom)
    )
    model = semi_amp * (
        cosw[None, :] * (ecc[None, :] + cosf) - sinw[None, :] * sinf
    )
    model += sigma * np.random.randn(*model.shape)

    sample_variance = np.zeros(N)
    for i, n in enumerate(rv_nb_transits):
        sample_variance[i] = np.var(model[:n, i], ddof=1)

    return rv_nb_transits, sample_variance


true_sigma = 1.0
binary_fraction = np.linspace(0, 0.5, 10)
datasets = []
for bf in binary_fraction:
    np.random.seed(251986)
    data = generate_dataset(true_sigma=true_sigma, binary_fraction=bf)
    datasets.append(data)


# +
def model(num_transit, sample_variance):
    log_sigma0 = numpyro.sample("log_sigma0", dist.Normal(0.0, 100.0))
    log_dsigma = numpyro.sample(
        "log_dsigma",
        dist.Normal(0.0, 100.0),
        sample_shape=(len(sample_variance),),
    )
    sigma2 = jnp.exp(2 * log_sigma0) + jnp.exp(2 * log_dsigma)
    stat = sample_variance * (num_transit - 1) / sigma2
    numpyro.sample("obs", dist.Chi2(num_transit + 1), obs=stat)


def guide(num_transit, sample_variance):
    mu_log_sigma0 = numpyro.param(
        "mu_log_sigma0", 0.5 * np.log(np.median(sample_variance))
    )
    sigma_log_sigma0 = numpyro.param(
        "sigma_log_sigma0", 1.0, constraint=dist.constraints.positive
    )

    mu_log_dsigma = numpyro.param(
        "mu_log_dsigma", 0.5 * np.log(sample_variance)
    )
    sigma_log_dsigma = numpyro.param(
        "sigma_log_dsigma",
        np.ones_like(sample_variance),
        constraint=dist.constraints.positive,
    )

    numpyro.sample("log_sigma0", dist.Normal(mu_log_sigma0, sigma_log_sigma0))
    numpyro.sample("log_dsigma", dist.Normal(mu_log_dsigma, sigma_log_dsigma))


optimizer = numpyro.optim.Adam(step_size=0.1)
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
# -

params = []
for n, data in enumerate(datasets):
    svi_result = svi.run(random.PRNGKey(1233 + n), 5000, *data)
    params.append(svi_result.params)

# +
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
fig.subplots_adjust(wspace=0.07)

ax = axes[0]
x = np.linspace(-2, 4, 500)
bins = np.linspace(-2, 4, 15)
for n, (bf, data) in enumerate(zip(binary_fraction, datasets)):
    ax.hist(
        np.log10(data[1]),
        bins,
        color=mpl.cm.viridis(1 - bf / binary_fraction.max()),
        lw=1,
        histtype="step",
        zorder=n,
    )
#     kde = stats.gaussian_kde(np.log10(data[1]))
#     ax.plot(x, len(data[1]) * kde(x) * (bins[1] - bins[0]), lw=0.5, color=mpl.cm.viridis(1 - bf / binary_fraction.max()), zorder=n)
ax.set_yscale("log")
ax.set_ylim(1, 1e3)
ax.set_xlim(x.min(), x.max())
ax.set_xlabel(r"measured, $\log_{10} s^2$  [km$^2$/s$^2$]")
ax.set_ylabel("count")
ax.set_title("simulated rv errors")

ax = axes[1]
x = np.log(true_sigma) + np.linspace(-0.1, 0.1, 500)
xp = np.log10(np.exp(x))
for bf, p in zip(binary_fraction, params):
    mean, std = p["mu_log_sigma0"], p["sigma_log_sigma0"]
    ax.plot(
        xp,
        np.exp(-0.5 * ((x - mean) / std) ** 2) / np.sqrt(2 * np.pi * std ** 2),
        lw=1,
        color=mpl.cm.viridis(1 - bf / binary_fraction.max()),
    )
ax.axvline(np.log10(true_sigma), lw=3.0, alpha=0.2, color="k")
ax.set_yticks([])
ax.set_xlabel(r"$\log_{10} \sigma_\mathrm{rv}$ [km/s]")
ax.set_xlim(xp.min(), xp.max())
ax.set_title("inferred per-transit uncertainty")

fig.savefig("sim_rv_uncertainty.pdf", bbox_inches="tight")
# -
