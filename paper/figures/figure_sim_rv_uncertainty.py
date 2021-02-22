# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from jax import random
import kepler

from one_datum.noise_model import setup_model


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


svi = setup_model()
params = []
for n, data in enumerate(datasets):
    svi_result = svi.run(random.PRNGKey(1233 + n), 5000, *data)
    params.append(svi_result.params)

# +
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(wspace=0.07)

ax = axes[0]
x = np.linspace(-2, 4, 500)
bins = np.linspace(-2, 4, 20)
for n, (bf, data) in enumerate(zip(binary_fraction, datasets)):
    ax.hist(
        np.log10(data[1]),
        bins,
        color=mpl.cm.viridis(1 - bf / binary_fraction.max()),
        lw=1,
        histtype="step",
        zorder=n,
    )
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
