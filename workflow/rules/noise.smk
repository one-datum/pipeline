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

rule postprocess_noise_model:
    input:
        expand("resources/data/noise-model{suffix}.fits", suffix=config["noise"]["suffix"])
    output:
        expand("src/one_datum/data/noise-model{suffix}.fits", suffix=config["noise"]["suffix"])
    conda:
        "../envs/noise.yaml"
    log:
        "results/logs/postprocess-noise-model.log"
    script:
        "../scripts/postprocess_noise_model.py"
