rule noise_infer:
    input:
        get_results_filename(config["base_table_filename"])
    output:
        get_results_filename("{dataset}/noise/raw.fits")
    params:
        config=config["noise_config_file"],
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
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/noise/postprocess.log")
    shell:
        """
        python workflow/scripts/noise/postprocess.py \\
            --input {input} --output {output} \\
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
