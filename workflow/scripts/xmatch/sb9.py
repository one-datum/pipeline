#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import re

import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import numpy as np
from astropy.io import ascii
from astroquery.gaia import Gaia

parser = argparse.ArgumentParser()
parser.add_argument("--readme", required=True, type=str)
parser.add_argument("--main", required=True, type=str)
parser.add_argument("--orbits", required=True, type=str)
parser.add_argument("--output", required=True, type=str)
parser.add_argument("--gaia-creds", required=True, type=str)
args = parser.parse_args()

# Log into the Gaia archive
gaia_creds = json.loads(args.gaia_creds)
Gaia.login(**gaia_creds)

# data_dir = pathlib.Path(args.input).parent
username = Gaia._TapPlus__user
sb9_tblname = "sb9_2021_04"
sb9_archive_tblname = f"user_{username}.{sb9_tblname}"

reader = ascii.get_reader(ascii.Cds, readme=args.readme)
sb9_tbl = reader.read(args.main)

# only keep sources with magnitudes in SB9 (this removes ~90 sources)
sb9_tbl = sb9_tbl[~sb9_tbl["mag1"].mask]

# also remove whack values - this removes another 2 sources
mag_mask = np.ones(len(sb9_tbl), dtype=bool)
for i, m in enumerate(sb9_tbl["mag1"]):
    try:
        float(m)
        mag_mask[i] = False
    except Exception:
        pass
sb9_tbl = sb9_tbl[~mag_mask]
sb9_tbl["mag1"] = sb9_tbl["mag1"].filled(np.nan).astype(float)
sb9_tbl["mag2"] = sb9_tbl["mag2"].filled(1e5)  # MAGIC NUMBER

ra = coord.Longitude(
    sb9_tbl["RAh"].data
    + sb9_tbl["RAm"].data / 60.0
    + sb9_tbl["RAs"].data / 3600.0,
    unit=u.hourangle,
)
sgn = 1 - 2 * (sb9_tbl["DE-"] == "-")
dec = coord.Latitude(
    sgn
    * (
        sb9_tbl["DEd"].data
        + sb9_tbl["DEm"].data / 60.0
        + sb9_tbl["DEs"].data / 3600.0
    ),
    unit=u.degree,
)

for k in list(sb9_tbl.columns):
    if k.startswith("RA") or k.startswith("DE"):
        del sb9_tbl[k]
    else:
        sb9_tbl[k].name = f"sb9_{k.lower()}"

sb9_tbl["sb9_seq"].name = "id"
sb9_tbl["ra_deg"] = ra.degree
sb9_tbl["dec_deg"] = dec.degree

tblnames = [t.name for t in Gaia.load_tables(only_names=True)]
if sb9_archive_tblname not in tblnames:
    job = Gaia.upload_table(
        sb9_tbl["id", "ra_deg", "dec_deg"], table_name=sb9_tblname
    )

# For subqueries, we can't use a wildcard to get all column names,
# so we explicitly have to ask for all gaia columns...
gaia_tbl = Gaia.load_table("gaiaedr3.gaia_source")
all_gaia_columns = [f"subq.{c.name}" for c in gaia_tbl.columns]

init_tol = 30 * u.arcsec
final_tol = 4 * u.arcsec
sb9_epoch = 2000.0  # This is a guess!

q = f"""
SELECT subq.id, {', '.join(all_gaia_columns)}
FROM (
    SELECT sb9.id, sb9.ra_deg, sb9.dec_deg, gaia.*
    FROM {sb9_archive_tblname} AS sb9, gaiaedr3.gaia_source as gaia
    WHERE
        contains(POINT('ICRS', sb9.ra_deg, sb9.dec_deg),
                 CIRCLE('ICRS', gaia.ra, gaia.dec, {init_tol.to_value(u.deg)}))=1
    OFFSET 0
) AS subq
WHERE
    contains(POINT('ICRS', subq.ra + subq.pmra / 3600e3  * ({sb9_epoch} - subq.ref_epoch) / COS(RADIANS(subq.dec)),
                           subq.dec + subq.pmdec / 3600e3 * ({sb9_epoch} - subq.ref_epoch)),
             CIRCLE('ICRS', subq.ra_deg, subq.dec_deg, {final_tol.to_value(u.deg)}))=1
"""  # noqa

job = Gaia.launch_job_async(q, name="sb9-edr3")
xm_tbl = job.get_results()
joined = at.join(sb9_tbl, xm_tbl, keys="id")

# OK so there are definitely sources with unexpected Gaia mags given the V-band
# mags in SB9, so we can trim those as (hopefully) duplicate / spurious matches.
# We could do something much better here, since there is a color-dependent
# transform between G <--> V...but, ya'know, hacks:
mag = -np.log10((10 ** -joined["sb9_mag1"] + 10 ** -joined["sb9_mag2"]))
dmag = mag - joined["phot_g_mean_mag"]
dmag_cut = 0.5
joined_dmag = joined[np.abs(dmag - np.median(dmag)) < dmag_cut]

# Now let's just save the ones with measured RV error from Gaia:
joined_dmag_with_rv = joined_dmag[
    np.isfinite(joined_dmag["dr2_radial_velocity_error"])
]
print(f"Found {len(joined_dmag_with_rv)} valid matches with Gaia RV")

orbit_tbl = reader.read(args.orbits)
for k in list(orbit_tbl.columns):
    orbit_tbl[k].name = f"sb9_{k.lower()}"
orbit_tbl["sb9_seq"].name = "id"

# And crossmatch
final = at.join(joined_dmag_with_rv, orbit_tbl, keys=["id"])
del final["designation"]

k1 = final["sb9_k1"]
k1[k1 == "12.D0"] = "12.0"
final["sb9_k1"] = k1.filled(np.nan).astype(float)
final.write(args.output, format="fits", overwrite=True)
