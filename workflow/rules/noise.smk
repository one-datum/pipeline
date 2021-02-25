rule infer_noise:
    input:
        "resources/data/edr3-rv-good-plx-result.fits.gz"
    output:
        expand("resources/data/noise-model{suffix}.fits", suffix=config["noise"]["suffix"])
    conda:
        "../envs/environment.yml"
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
        "../envs/environment.yml"
    log:
        "results/logs/postprocess-noise-model.log"
    script:
        "../scripts/postprocess_noise_model.py"

rule apply_noise_model:
    input:
        "resources/data/edr3-rv-good-plx-result.fits.gz"
        expand("src/one_datum/data/noise-model{suffix}.fits", suffix=config["noise"]["suffix"])
    output:
        expand("resources/data/edr3-rv-good-plx-plus-semiamp{suffix}.fits.gz", suffix=config["noise"]["suffix"])
    conda:
        "../envs/environment.yml"
    log:
        "results/logs/apply-noise-model.log"
    script:
        "../scripts/apply_noise_model.py"
