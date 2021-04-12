# Rules to infer the noise model

rule infer_noise:
    input:
        install=get_results_filename("install.done"),
        base_table=get_remote_filename(config["base_table_filename"])
    output:
        get_results_filename("{dataset}/noise/raw-noise-model.fits")
    params:
        config=config["noise_config_file"],
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/noise/infer-noise.log")
    shell:
        """
        python workflow/scripts/infer_noise.py \\
            --input {input.base_table} \\
            --output {output} \\
            --config {params.config} \\
            &> {log}
        """

rule postprocess_noise_model:
    input:
        get_results_filename("{dataset}/noise/raw-noise-model.fits")
    output:
        get_results_filename("{dataset}/noise/smoothed-noise-model.fits")
    params:
        color_smooth=config["noise"]["color_smoothing_scale"],
        mag_smooth=config["noise"]["mag_smoothing_scale"]
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/noise/postprocess-noise-model.log")
    shell:
        """
        python workflow/scripts/postprocess_noise_model.py \\
            --input {input} --output {output} \\
            --color-smooth {params.color_smooth} \\
            --mag-smooth {params.mag_smooth} \\
            &> {log}
        """

rule apply_noise_model:
    input:
        noise_model=get_results_filename("{dataset}/noise/smoothed-noise-model.fits"),
        base_table=get_remote_filename(config["base_table_filename"])
    output:
        get_results_filename("{dataset}/noise/estimated.fits.gz")
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/noise/apply-noise-model.log")
    shell:
        """
        python workflow/scripts/apply_noise_model.py \\
            --noise-model {input.noise_model} \\
            --input {input.base_table} \\
            --output {output} \\
            &> {log}
        """
