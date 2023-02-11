import os

URLS = {
    # "gold_sample.fits": "https://users.flatironinstitute.org/~apricewhelan/data/dr16-binaries/gold_sample.fits",
    "gold_sample.fits": "https://users.flatironinstitute.org/~apricewhelan/data/apogee/clean-unimodal-turbo20-beta.fits",
    "kepler_dr2_1arcsec.fits": "https://www.dropbox.com/s/xo1n12fxzgzybny/kepler_dr2_1arcsec.fits?dl=1",
    "kepler_edr3_1arcsec.fits": "https://www.dropbox.com/s/bkek5qc4hdnlz7f/kepler_edr3_1arcsec.fits?dl=0",
    # "base.fits.gz": "https://zenodo.org/record/7007600/files/one-datum-dr3-result.fits.gz?download=1",
}

localrules: get_data

rule get_data:
    output:
        get_remote_filename("{filename}")
    params:
        url=lambda wildcards: URLS[wildcards[0]]
    conda:
        "../envs/remote.yml"
    log:
        get_log_filename("remote/{filename}.log")
    shell:
        "curl -L \"{params.url}\" -o {output} 2> {log}"


if os.path.exists("/mnt/ceph/users/gaia/dr3/hdf5/GaiaSource"):
    rule build_base_dataset:
        output:
            get_remote_filename("base.fits.gz")
        conda:
            "../envs/munge.yml"
        log:
            get_log_filename("remote/base.fits.gz.log")
        shell:
            "python workflow/scripts/data.py {output} &> {log}"

else:
    rule download_base_dataset:
        output:
            get_remote_filename("base.fits.gz")
        params:
            url="https://zenodo.org/record/7007600/files/one-datum-dr3-result.fits.gz?download=1"
        conda:
            "../envs/remote.yml"
        log:
            get_log_filename("base.fits.gz.log")
        shell:
            "curl -L \"{params.url}\" -o {output} 2> {log}"
