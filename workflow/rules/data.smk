URLS = {
    "gold_sample.fits": "https://users.flatironinstitute.org/~apricewhelan/data/dr16-binaries/gold_sample.fits",
    "edr3-rv-good-plx-result.fits.gz": "https://users.flatironinstitute.org/~apricewhelan/data/edr3/edr3-rv-good-plx-result.fits.gz",
    "kepler_dr2_1arcsec.fits": "https://www.dropbox.com/s/xo1n12fxzgzybny/kepler_dr2_1arcsec.fits?dl=1",
}

rule get_data:
    output:
        "resources/data/{filename}"
    params:
        url=lambda wildcards: URLS[wildcards[0]]
    log:
        "results/logs/get-data-{filename}.log"
    run:
        shell("curl -L \"{params.url}\" -o {output} 2> {log}")
