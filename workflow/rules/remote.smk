URLS = {
    "gold_sample.fits": "https://users.flatironinstitute.org/~apricewhelan/data/dr16-binaries/gold_sample.fits",
    "edr3-rv-good-plx-result.fits.gz": "https://users.flatironinstitute.org/~apricewhelan/data/edr3/edr3-rv-good-plx-result.fits.gz",
    "kepler_dr2_1arcsec.fits": "https://www.dropbox.com/s/xo1n12fxzgzybny/kepler_dr2_1arcsec.fits?dl=1",
    "kepler_edr3_1arcsec.fits": "https://www.dropbox.com/s/bkek5qc4hdnlz7f/kepler_edr3_1arcsec.fits?dl=0",
    "SB9public.tar.gz": "https://sb9.astro.ulb.ac.be/SB9public.tar.gz",
    "sb9/orbits.dat.gz": "ftp://cdsarc.u-strasbg.fr/pub/cats/B/sb9/orbits.dat.gz",
    "sb9/main.dat.gz": "ftp://cdsarc.u-strasbg.fr/pub/cats/B/sb9/main.dat.gz",
    "sb9/ReadMe": "ftp://cdsarc.u-strasbg.fr/pub/cats/B/sb9/ReadMe",
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
