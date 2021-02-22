# -*- coding: utf-8 -*-

__all__ = ["setup_model"]

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.config import config
from numpyro.infer import SVI, Trace_ELBO

config.update("jax_enable_x64", True)


def setup_model() -> SVI:
    def model(num_transit, sample_variance):
        log_sigma0 = numpyro.sample("log_sigma0", dist.Normal(0.0, 10.0))
        log_dsigma = numpyro.sample(
            "log_dsigma",
            dist.Normal(0.0, 10.0),
            sample_shape=(len(sample_variance),),
        )
        sigma2 = jnp.exp(2 * log_sigma0) + jnp.exp(2 * log_dsigma)
        stat = sample_variance * (num_transit - 1)
        numpyro.sample(
            "obs", dist.Gamma(0.5 * (num_transit - 1), 0.5 / sigma2), obs=stat
        )

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

        numpyro.sample(
            "log_sigma0", dist.Normal(mu_log_sigma0, sigma_log_sigma0)
        )
        numpyro.sample(
            "log_dsigma", dist.Normal(mu_log_dsigma, sigma_log_dsigma)
        )

    optimizer = numpyro.optim.Adam(step_size=0.05)
    return SVI(model, guide, optimizer, loss=Trace_ELBO())
