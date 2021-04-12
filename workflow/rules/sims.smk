rule simulate_catalog:
    output:
        get_results_filename("{dataset}/sims/simulated.fits.gz")
    params:
        config=config["sims_config_file"],
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/sims/simulate-catalog.log")
    shell:
        """
        python workflow/scripts/simulate_catalog.py \\
            --output {output} \\
            --config {params.config} \\
            &> {log}
        """