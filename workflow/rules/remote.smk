URLS = {
    # "gold_sample.fits": "https://users.flatironinstitute.org/~apricewhelan/data/dr16-binaries/gold_sample.fits",
    "gold_sample.fits": "https://users.flatironinstitute.org/~apricewhelan/data/apogee/clean-unimodal-turbo20-beta.fits",
    "kepler_dr2_1arcsec.fits": "https://www.dropbox.com/s/xo1n12fxzgzybny/kepler_dr2_1arcsec.fits?dl=1",
    "kepler_edr3_1arcsec.fits": "https://www.dropbox.com/s/bkek5qc4hdnlz7f/kepler_edr3_1arcsec.fits?dl=0",
}

localrules: get_data

rule get_data:
    output:
        get_remote_filename("{filename}")
    params:
        url=lambda wildcards: URLS[wildcards[0]]
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("remote/{filename}.log")
    shell:
        "curl -L \"{params.url}\" -o {output} 2> {log}"
