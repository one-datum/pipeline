#!/usr/bin/env bash

datadir=${1:-.}
wget https://users.flatironinstitute.org/~apricewhelan/data/edr3/edr3-rv-good-plx-result.fits.gz --directory-prefix=$datadir
wget https://users.flatironinstitute.org/~apricewhelan/data/dr16-binaries/gold_sample.fits --directory-prefix=$datadir
