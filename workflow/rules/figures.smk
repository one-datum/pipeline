FIGURES = [
    "completeness",
    "recovered",
    "noise_model",
    "binary_fraction_cmd",
]

rule figures_completeness:
    input:
        get_results_filename("{dataset}/simulations/inferred.fits.gz")
    output:
        report(
            get_results_filename("{dataset}/figures/completeness.png"),
            category="Simulations",
        )
    params:
        threshold=config["det_pval_thresh"]
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/figures/completeness.log")
    shell:
        """
        python workflow/scripts/figures/completeness.py \\
            --input {input} \\
            --output {output} \\
            --threshold {params.threshold} \\
            &> {log}
        """

rule figures_recovered:
    input:
        get_results_filename("{dataset}/simulations/inferred.fits.gz")
    output:
        report(
            get_results_filename("{dataset}/figures/recovered.png"),
            category="Simulations",
        )
    params:
        threshold=config["det_pval_thresh"]
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/figures/recovered.log")
    shell:
        """
        python workflow/scripts/figures/recovered.py \\
            --input {input} \\
            --output {output} \\
            --threshold {params.threshold} \\
            &> {log}
        """

rule figures_noise_model:
    input:
        get_results_filename("{dataset}/noise/processed.fits")
    output:
        report(
            get_results_filename("{dataset}/figures/noise_model.png"),
            category="Noise model",
        )
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/figures/noise_model.log")
    shell:
        """
        python workflow/scripts/figures/noise_model.py \\
            --input {input} \\
            --output {output} \\
            &> {log}
        """

rule figures_binary_fraction_cmd:
    input:
        get_results_filename("{dataset}/inference/inferred.fits.gz")
    output:
        report(
            get_results_filename("{dataset}/figures/binary_fraction_cmd.png"),
            category="Catalog",
        )
    params:
        threshold=config["det_pval_thresh"]
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/figures/binary_fraction_cmd.log")
    shell:
        """
        python workflow/scripts/figures/binary_fraction_cmd.py \\
            --input {input} \\
            --output {output} \\
            --threshold {params.threshold} \\
            &> {log}
        """
