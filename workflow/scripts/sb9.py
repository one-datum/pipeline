#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json

import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, type=str)
parser.add_argument("--figure", required=True, type=str)
parser.add_argument("--gaia-creds", required=True, type=str)
args = parser.parse_args()

# Log into the Gaia archive
gaia_creds = json.loads(args.gaia_creds)
Gaia.login(**gaia_creds)

# data_dir = pathlib.Path(args.input).parent
username = Gaia._TapPlus__user
sb9_tblname = "sb9_2021_04"
sb9_archive_tblname = f"user_{username}.{sb9_tblname}"

# Download the tables from Vizier
v = Vizier(columns=["*", "+_r"], catalog="sb9/main")
v.ROW_LIMIT = -1
result = v.query_constraints()
sb9_tbl = result[0]

v = Vizier(columns=["*", "+_r"], catalog="sb9/orbits")
v.ROW_LIMIT = -1
result = v.query_constraints()
orbit_tbl = result[0]

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

ra = coord.Angle(sb9_tbl["RAJ2000"], u.hourangle)
dec = coord.Angle(sb9_tbl["DEJ2000"], u.degree)

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
# mags in SB9, so we can trim those as (hopefully) duplicate / spurious
# matches. We could do something much better here, since there is a
# color-dependent transform between G <--> V...but, ya'know, hacks:
mag = -np.log10((10 ** -joined["sb9_mag1"] + 10 ** -joined["sb9_mag2"]))
dmag = mag - joined["phot_g_mean_mag"]
dmag_cut = 0.5
mask = np.abs(dmag - np.median(dmag)) < dmag_cut
joined_dmag = joined[mask]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 6))
ax = axes[0]
ax.scatter(
    joined["phot_g_mean_mag"],
    dmag,
    alpha=0.2,
    lw=0,
    s=4,
    c="k",
)
ax.scatter(
    joined["phot_g_mean_mag"][mask],
    dmag[mask],
    alpha=0.5,
    lw=0,
    s=4,
    c="C1",
    label="selected",
)
ax.set_xlabel("$G$ [mag]")
ax.set_ylabel(r"$V_{\rm SB9} - G$ [mag]")
ax.set_ylim(-5, 5)
ax.legend()

# Now let's just save the ones with measured RV error from Gaia:
joined_dmag_with_rv = joined_dmag[
    np.isfinite(joined_dmag["dr2_radial_velocity_error"])
]
print(f"Found {len(joined_dmag)} matched; {len(joined_dmag_with_rv)} with RVs")

for k in list(orbit_tbl.columns):
    orbit_tbl[k].name = f"sb9_{k.lower()}"
orbit_tbl["sb9_seq"].name = "id"

# And crossmatch
final = at.join(joined_dmag_with_rv, orbit_tbl, keys=["id"])
del final["designation"]

final["sb9_k1"] = final["sb9_k1"].filled(np.nan).astype(float)
final.write(args.output, format="fits", overwrite=True)

ax = axes[1]
ax.loglog(final["sb9_k1"], final["dr2_radial_velocity_error"], ".")
ax.set_xlabel("SB9 K1")
ax.set_ylabel("Gaia DR2 RV error")
ax.set_title(f"{len(final)} total matches")
fig.savefig(args.figure, bbox_inches="tight")
