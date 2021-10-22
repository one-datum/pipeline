rule simulations_catalog:
    output:
        get_results_filename("simulations/catalog.fits.gz")
    params:
        config=config["sims_config_file"],
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("simulations/catalog.log")
    shell:
        """
        python workflow/scripts/simulate.py \\
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
        "../envs/environment.yml"
    log:
        get_log_filename("simulations/inference.log")
    shell:
        """
        python workflow/scripts/inference.py \\
            --input {input} \\
            --output {output} \\
            &> {log}
        """
