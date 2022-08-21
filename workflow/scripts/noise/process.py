import argparse
import pickle

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from astropy.io import fits
from tinygp import GaussianProcess, kernels, transforms

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True, type=str)
parser.add_argument("--output-grid", required=True, type=str)
parser.add_argument("--output-gp", required=True, type=str)
parser.add_argument("--color-smooth", required=True, type=float)
parser.add_argument("--mag-smooth", required=True, type=float)
args = parser.parse_args()

jax.config.update("jax_enable_x64", True)

# Load the data file
with fits.open(args.input) as f:
    hdr = f[0].header
    data = f[1].data
    count = f[2].data

# Compute the bin coordinates from the header specification
color_bins = np.linspace(hdr["MIN_COL"], hdr["MAX_COL"], hdr["NUM_COL"] + 1)
mag_bins = np.linspace(hdr["MIN_MAG"], hdr["MAX_MAG"], hdr["NUM_MAG"] + 1)
color_bin_centers = 0.5 * (color_bins[1:] + color_bins[:-1])
mag_bin_centers = 0.5 * (mag_bins[1:] + mag_bins[:-1])

# Set up the data grid for the GP interpolation
x_, y_ = color_bin_centers, mag_bin_centers
X_, Y_ = np.meshgrid(x_, y_, indexing="ij")
y = np.ascontiguousarray(np.mean(np.log(data), axis=-1).T).flatten()
yerr = np.ascontiguousarray(np.std(np.log(data), axis=-1).T).flatten()
valid = np.isfinite(y)
X = np.stack((X_.flatten()[valid], Y_.flatten()[valid]), axis=-1)
y = y[valid]
yerr = yerr[valid]

# Set up the GP model
def build_gp(params):
    kernel = jnp.exp(params["log_amp"]) * transforms.Cholesky.from_parameters(
        jnp.exp(params["log_scale"][:2]),
        params["log_scale"][2:],
        kernels.ExpSquared(),
    )
    return GaussianProcess(
        kernel,
        X,
        diag=yerr**2 + jnp.exp(2 * params["log_jitter"]),
        mean=params["mean"],
    )


@jax.jit
def get_pred(params):
    cond = build_gp(params).condition(
        y, np.stack((X_.flatten(), Y_.flatten()), axis=-1)
    )
    pred = cond.gp.loc.reshape(X_.shape)
    pred_std = jnp.sqrt(cond.gp.variance).reshape(X_.shape)
    return pred.reshape(X_.shape), pred_std.reshape(X_.shape)


@jax.jit
def loss(params):
    return -build_gp(params).log_probability(y)


# Run the optimization
init = {
    "log_amp": jnp.log(0.1),
    "log_scale": jnp.log(jnp.full(3, 0.1)),
    "log_jitter": jnp.log(1e-4),
    "mean": 0.0,
}
opt = jaxopt.ScipyMinimize(fun=loss)
soln = opt.run(init)
pred, pred_std = get_pred(soln.params)
valid = valid.reshape(X_.shape)

# Save the results
hdr["col_smth"] = args.color_smooth
hdr["mag_smth"] = args.mag_smooth
fits.HDUList(
    [
        fits.PrimaryHDU(header=hdr),
        fits.ImageHDU(pred),
        fits.ImageHDU(pred_std),
        fits.ImageHDU(valid.astype(np.int32)),
        fits.ImageHDU(count),
        fits.ImageHDU(data),
    ]
).writeto(args.output_grid, overwrite=True)

with open(args.output_gp, "wb") as f:
    pickle.dump((soln.params, y, build_gp(soln.params)), f)
