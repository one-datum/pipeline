rule simulations_catalog:
    input:
        get_remote_filename(config["base_table_filename"])
    output:
        get_results_filename("simulations/catalog.fits.gz")
    params:
        config=config["sims_config_file"],
    conda:
        "../envs/baseline.yml"
    log:
        get_log_filename("simulations/catalog.log")
    shell:
        """
        python workflow/scripts/simulate.py \\
            --input {input} \\
            --output {output} \\
            --config {params.config} \\
            &> {log}
        """

rule simulations_inference:
    input:
        get_results_filename("simulations/catalog.fits.gz")
    output:
        get_results_filename("simulations/inferred.fits.gz")
    conda:
        "../envs/inference.yml"
    log:
        get_log_filename("simulations/inference.log")
    shell:
        """
        python workflow/scripts/inference.py \\
            --input {input} \\
            --output {output} \\
            &> {log}
        """
