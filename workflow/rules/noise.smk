rule noise_infer:
    input:
        get_results_filename(config["base_table_filename"])
    output:
        get_results_filename("{dataset}/noise/raw.fits")
    params:
        config=config["noise_config_file"]
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/noise/infer.log")
    shell:
        """
        python workflow/scripts/noise/infer.py \\
            --input {input} \\
            --output {output} \\
            --config {params.config} \\
            &> {log}
        """

rule noise_postprocess:
    input:
        get_results_filename("{dataset}/noise/raw.fits")
    output:
        get_results_filename("{dataset}/noise/processed.fits")
    params:
        color_smooth=config["noise"]["color_smoothing_scale"],
        mag_smooth=config["noise"]["mag_smoothing_scale"]
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/noise/postprocess.log")
    shell:
        """
        python workflow/scripts/noise/postprocess.py \\
            --input {input} --output {output} \\
            --color-smooth {params.color_smooth} \\
            --mag-smooth {params.mag_smooth} \\
            &> {log}
        """

rule noise_apply:
    input:
        noise_model=get_results_filename("{dataset}/noise/processed.fits"),
        base_table=get_results_filename(config["base_table_filename"])
    output:
        get_results_filename("{dataset}/noise/applied.fits.gz")
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/noise/apply.log")
    shell:
        """
        python workflow/scripts/noise/apply.py \\
            --noise-model {input.noise_model} \\
            --input {input.base_table} \\
            --output {output} \\
            &> {log}
        """

rule noise_calibrate:
    input:
        get_results_filename("{dataset}/noise/applied.fits.gz")
    output:
        get_results_filename("{dataset}/noise/calibrate.txt")
    params:
        config=config["noise_config_file"]
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/noise/calibrate.log")
    shell:
        """
        python workflow/scripts/noise/calibrate.py \\
            --input {input} \\
            --output {output} \\
            --config {params.config} \\
            &> {log}
        """

rule noise_pval:
    input:
        catalog=get_results_filename("{dataset}/noise/applied.fits.gz"),
        calib=get_results_filename("{dataset}/noise/calibrate.txt")
    output:
        get_results_filename("{dataset}/noise/pval.fits.gz")
    params:
        config=config["noise_config_file"]
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/noise/pval.log")
    shell:
        """
        python workflow/scripts/noise/pval.py \\
            --catalog {input.catalog} \\
            --calib {input.calib} \\
            --output {output} \\
            --config {params.config} \\
            &> {log}
        """
