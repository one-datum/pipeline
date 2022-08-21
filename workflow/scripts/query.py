"""Note: this script is not used because of instability of the Gaia archive. The
query results are downloaded from Zenodo in rules/remote.smk now.
"""

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
    phot_g_mean_mag,radial_velocity_error,rv_nb_transits,
	rv_method_used,rv_visibility_periods_used,
	rv_renormalised_gof,rv_chisq_pvalue,rv_time_duration,
	rv_amplitude_robust,rv_template_teff,rv_template_logg,
	rv_template_fe_h
FROM gaiadr3.gaia_source
WHERE radial_velocity_error IS NOT NULL
AND parallax_over_error > 4
AND rv_nb_transits > 3
AND rv_method_used = 1
"""
job = Gaia.launch_job_async(q, name="one-datum-dr3")
tbl = job.get_results()
tbl.write(args.output, format="fits", overwrite=True)
