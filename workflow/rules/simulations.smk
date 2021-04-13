rule simulate_catalog:
    output:
        get_results_filename("{dataset}/simulations/simulated.fits.gz")
    params:
        config=config["sims_config_file"],
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/simulations/simulate-catalog.log")
    shell:
        """
        python workflow/scripts/simulate_catalog.py \\
            --output {output} \\
            --config {params.config} \\
            &> {log}
        """

rule bulk_inference_sims:
    input:
        get_results_filename("{dataset}/simulations/simulated.fits.gz")
    output:
        get_results_filename("{dataset}/simulations/processed.fits.gz")
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/simulations/bulk-inference.log")
    shell:
        """
        python workflow/scripts/bulk_inference.py \\
            --input {input} \\
            --output {output} \\
            &> {log}
        """

rule completeness_plots:
    input:
        get_results_filename("{dataset}/simulations/processed.fits.gz")
    output:
        report(
            directory(get_results_filename("{dataset}/simulations/completeness")),
            patterns=["{name}.png", "{name}.pdf"],
            category="Completeness",
        )
    params:
        threshold=config["det_pval_thresh"],
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/simulations/completeness.log")
    shell:
        """
        python workflow/scripts/completeness.py \\
            --input {input} \\
            --output {output} \\
            --threshold {params.threshold} \\
            &> {log}
        """
