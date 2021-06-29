#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys

import astropy.table as at
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--reference", required=True, type=str)
parser.add_argument("--table", required=True, type=str)
parser.add_argument("--output", required=True, type=str)
parser.add_argument("--source-id-col", type=str)
parser.add_argument("--kcol", type=str)
parser.add_argument("--kerrcol", type=str)
parser.add_argument("--figure", type=str)
parser.add_argument("--threshold", type=str)
parser.add_argument("--name", type=str, default="catalog")
args = parser.parse_args()

reference = at.Table.read(args.reference)
if args.source_id_col:
    reference["source_id"] = reference[args.source_id_col]
table = at.Table.read(args.table)
joined = at.join(reference, table, keys=["source_id"])
joined.write(args.output, format="fits", overwrite=True)

if not (args.kcol and args.figure):
    sys.exit(0)

k = joined[args.kcol]
if args.kerrcol:
    kerr = joined[args.kerrcol]
else:
    kerr = np.zeros(len(k))

y = joined["rv_semiamp_p50"]
yerr = (
    joined["rv_semiamp_p50"] - joined["rv_semiamp_p16"],
    joined["rv_semiamp_p84"] - joined["rv_semiamp_p50"],
)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

ax = axes[0]
chi = y - k
chi[chi >= 0] /= yerr[0][chi >= 0]
chi[chi < 0] /= yerr[1][chi < 0]
ax.hist(chi, 50, range=(-10, 10), color="k", histtype="step", density=True)
x0 = np.linspace(-10, 10, 500)
ax.plot(x0, np.exp(-(x0 ** 2)) / np.sqrt(2 * np.pi), color="C1", lw=0.5)
ax.set_yticks([])
ax.set_xlabel("normalized residuals")

ax = axes[1]
ax.errorbar(k, y, xerr=kerr, yerr=yerr, fmt=",k", alpha=0.4, lw=0.25)
c = ax.scatter(
    k,
    y,
    c=joined["rv_pval"],
    s=5,
    zorder=10,
    vmin=0,
    vmax=args.threshold,
    edgecolor="none",
    cmap="inferno",
)
ax.set_xscale("log")
ax.set_yscale("log")

xlim = ax.get_xlim()
ylim = ax.get_ylim()
lim = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
ax.plot(lim, lim, "k", lw=0.5, alpha=0.8)
ax.set_xlim(lim)
ax.set_ylim(lim)

ax.set_xlabel(f"{args.name} semi-amplitude [km/s]")
ax.set_ylabel("inferred semi-amplitude [km/s]")

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.854, 0.125, 0.02, 0.75])
fig.colorbar(c, label="p-value", cax=cbar_ax)

fig.savefig(args.figure, bbox_inches="tight")
