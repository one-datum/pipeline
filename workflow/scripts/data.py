import glob
import sys

import h5py
import numpy as np
import tqdm
from astropy.table import Table, vstack

output_filename = sys.argv[1]

columns = [
    "source_id",
    "ra",
    "dec",
    "parallax",
    "parallax_over_error",
    "bp_rp",
    "phot_g_mean_mag",
    "radial_velocity_error",
    "rv_nb_transits",
    "rv_method_used",
    "rv_visibility_periods_used",
    "rv_renormalised_gof",
    "rv_chisq_pvalue",
    "rv_time_duration",
    "rv_amplitude_robust",
    "rv_template_teff",
    "rv_template_logg",
    "rv_template_fe_h",
]
data = None

for n, fn in enumerate(
    tqdm.tqdm(sorted(glob.glob("/mnt/ceph/users/gaia/dr3/hdf5/GaiaSource/*")))
):
    with h5py.File(fn, "r") as f:
        select = np.isfinite(f["radial_velocity_error"][...])
        select &= f["parallax_over_error"][...] > 4.0
        select &= f["rv_nb_transits"][...] > 3
        select &= f["phot_g_mean_mag"][...] < 16

        subdata = Table()
        for k in columns:
            subdata[k] = f[k][...][select]

        if data is None:
            data = subdata
        else:
            data = vstack([data, subdata])

data.write(output_filename, overwrite=True)
