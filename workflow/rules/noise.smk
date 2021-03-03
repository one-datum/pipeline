# Rules to infer the noise model

rule infer_noise:
    input:
        config["noise"]["base_table_filename"],
        "results/installed.done"
    output:
        "results/noise/{dataset}/raw-noise-model.fits"
    conda:
        "../envs/environment.yml"
    log:
        "results/logs/noise/{dataset}/infer-noise.log"
    script:
        "../scripts/infer_noise.py"

rule postprocess_noise_model:
    input:
        "results/noise/{dataset}/raw-noise-model.fits"
    output:
        "results/noise/{dataset}/smoothed-noise-model.fits"
    conda:
        "../envs/environment.yml"
    log:
        "results/logs/noise/{dataset}/postprocess-noise-model.log"
    script:
        "../scripts/postprocess_noise_model.py"

rule install_noise_model:
    input:
        "results/noise/{dataset}/smoothed-noise-model.fits"
    output:
        "src/one_datum/data/{dataset}-noise-model.fits"
    log:
        "results/logs/noise/{dataset}/install-noise-model.log"
    shell:
        "cp {input} {output} &> {log}"
