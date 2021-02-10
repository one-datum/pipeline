URLS = {
    "gold_sample.fits": "https://users.flatironinstitute.org/~apricewhelan/data/dr16-binaries/gold_sample.fits",
    "edr3-rv-good-plx-result.fits.gz": "https://users.flatironinstitute.org/~apricewhelan/data/edr3/edr3-rv-good-plx-result.fits.gz",
    "kepler_dr2_1arcsec.fits": "https://www.dropbox.com/s/xo1n12fxzgzybny/kepler_dr2_1arcsec.fits?dl=1",
    "SB9public.tar.gz": "https://sb9.astro.ulb.ac.be/SB9public.tar.gz",
}

rule get_data:
    output:
        "resources/data/{filename}"
    params:
        url=lambda wildcards: URLS[wildcards[0]]
    log:
        "results/logs/get-data-{filename}.log"
    shell:
        "curl -L \"{params.url}\" -o {output} 2> {log}"

rule sb9_data:
    input:
        "resources/data/SB9public.tar.gz"
    output:
        directory("resources/data/SB9public")
    log:
        "results/logs/sb9-data.log"
    shell:
        "mkdir -p {output};tar -xzvf {input} -C {output} 2> {log}"

rule sb9_xmatch:
    input:
        "resources/data/SB9public"
    output:
        "resources/data/sb9-gaia-xmatch.fits"
    conda:
        "../envs/environment.yml"
    log:
        notebook="results/logs/sb9-gaia-xmatch.ipynb"
    notebook:
        "../notebooks/sb9-gaia-xmatch.py.ipynb"
