rule infer_noise:
    input:
        "resources/data/edr3-rv-good-plx-result.fits.gz"
    output:
        expand("resources/data/noise-model{suffix}.fits", suffix=config["noise"]["suffix"])
    conda:
        "../envs/noise.yaml"
    log:
        "results/logs/infer-noise.log"
    script:
        "../scripts/infer_noise.py"
