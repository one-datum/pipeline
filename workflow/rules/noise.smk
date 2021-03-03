# Rules to infer the noise model

rule infer_noise:
    input:
        config["noise"]["base_table_filename"],
        "results/installed.done"
    output:
        "results/{dataset}/noise/raw-noise-model.fits"
    conda:
        "../envs/environment.yml"
    log:
        "results/logs/{dataset}/noise/infer-noise.log"
    script:
        "../scripts/infer_noise.py"

rule postprocess_noise_model:
    input:
        "results/{dataset}/noise/raw-noise-model.fits"
    output:
        "results/{dataset}/noise/smoothed-noise-model.fits"
    conda:
        "../envs/environment.yml"
    log:
        "results/logs/{dataset}/noise/postprocess-noise-model.log"
    script:
        "../scripts/postprocess_noise_model.py"

rule install_noise_model:
    input:
        "results/{dataset}/noise/smoothed-noise-model.fits"
    output:
        "src/one_datum/data/{dataset}-noise-model.fits"
    log:
        "results/logs/{dataset}/noise/install-noise-model.log"
    shell:
        "cp {input} {output} &> {log}"
