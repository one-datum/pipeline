#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json

from astroquery.gaia import Gaia

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, type=str)
parser.add_argument("--gaia-creds", required=True, type=str)
args = parser.parse_args()


# Log into the Gaia archive
gaia_creds = json.loads(args.gaia_creds)
Gaia.login(**gaia_creds)

q = """
SELECT
    source_id,ra,dec,parallax,parallax_over_error,bp_rp,
    phot_g_mean_mag,dr2_radial_velocity_error,dr2_rv_nb_transits
FROM gaiaedr3.gaia_source
WHERE dr2_radial_velocity_error IS NOT NULL
AND parallax_over_error > 4
AND dr2_rv_nb_transits > 2
"""
job = Gaia.launch_job_async(q, name="one-datum")
tbl = job.get_results()
tbl.write(args.output, format="fits", overwrite=True)
