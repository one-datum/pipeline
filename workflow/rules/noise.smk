# Rules to infer the noise model

rule infer_noise:
    input:
        base_table=config["base_table_filename"],
        installed="results/installed.done"
    output:
        "results/{dataset}/noise/raw-noise-model.fits"
    params:
        config=config["noise_config_file"],
    conda:
        "../envs/environment.yml"
    log:
        "results/logs/{dataset}/noise/infer-noise.log"
    shell:
        "python workflow/scripts/infer_noise.py --input {input.base_table} --output {output} --config {params.config} &> {log}"

rule postprocess_noise_model:
    input:
        "results/{dataset}/noise/raw-noise-model.fits"
    output:
        "results/{dataset}/noise/smoothed-noise-model.fits"
    params:
        color_smooth=config["noise"]["color_smoothing_scale"],
        mag_smooth=config["noise"]["mag_smoothing_scale"]
    conda:
        "../envs/environment.yml"
    log:
        "results/logs/{dataset}/noise/postprocess-noise-model.log"
    shell:
        """
        python workflow/scripts/postprocess_noise_model.py \\
            --input {input} --output {output} \\
            --color-smooth {params.color_smooth} \\
            --mag-smooth {params.mag_smooth} \\
            &> {log}
        """

rule install_noise_model:
    input:
        "results/{dataset}/noise/smoothed-noise-model.fits"
    output:
        "src/one_datum/data/{dataset}-noise-model.fits"
    log:
        "results/logs/{dataset}/noise/install-noise-model.log"
    shell:
        "cp {input} {output} &> {log}"

rule apply_noise_model:
    input:
        base_table=config["base_table_filename"],
        installed="results/installed.done",
        grid=expand("src/one_datum/data/{base_dataset_name}-noise-model.fits", base_dataset_name=config["base_dataset_name"])
    output:
        "results/{dataset}/noise/estimated.fits.gz"
    conda:
        "../envs/environment.yml"
    log:
        "results/logs/{dataset}/noise/apply-noise-model.log"
    shell:
        "python workflow/scripts/apply_noise_model.py --input {input.base_table} --output {output} &> {log}"
