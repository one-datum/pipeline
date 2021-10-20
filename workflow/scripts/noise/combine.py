#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


def sorter(fn):
    return int(os.path.splitext(os.path.split(fn)[-1])[0].split("-")[-1])


if __name__ == "__main__":
    import argparse

    import numpy as np
    import yaml
    from astropy.io import fits

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-c", "--config", required=True, type=str)
    parser.add_argument("input", nargs="+")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    mu = []
    sigma = []
    count = []
    for inp in sorted(args.input, key=sorter):
        with fits.open(inp) as hdu:
            mu.append(hdu[1].data)
            sigma.append(hdu[2].data)
            count.append(hdu[3].data)
    mu = np.concatenate(mu, axis=0)
    sigma = np.concatenate(sigma, axis=0)
    count = np.concatenate(count, axis=0)

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
