FIGURES = [
    get_filename_for_dataset(f"figures/{figure}.pdf")
    for figure in [
        "completeness",
        "recovered",
        "noise_model",
        "sigma_cmd",
        "binary_fraction_cmd",
        "p_value_dist",
    ]
]

rule figures_completeness:
    input:
        get_results_filename("simulations/catalog.fits.gz")
    output:
        report(
            get_results_filename("figures/completeness.pdf"),
            category="Simulations",
        )
    params:
        threshold=config["det_pval_thresh"]
    conda:
        "../envs/figures.yml"
    log:
        get_log_filename("figures/completeness.log")
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
        get_results_filename("simulations/inferred.fits.gz")
    output:
        report(
            get_results_filename("figures/recovered.pdf"),
            category="Simulations",
        )
    params:
        threshold=config["det_pval_thresh"]
    conda:
        "../envs/figures.yml"
    log:
        get_log_filename("figures/recovered.log")
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
        get_results_filename("noise/process.fits")
    output:
        report(
            get_results_filename("figures/noise_model.pdf"),
            category="Noise model",
        )
    conda:
        "../envs/figures.yml"
    log:
        get_log_filename("figures/noise_model.log")
    shell:
        """
        python workflow/scripts/figures/noise_model.py \\
            --input {input} \\
            --output {output} \\
            &> {log}
        """

rule figures_sigma_cmd:
    input:
        get_results_filename("noise/apply.fits.gz")
    output:
        report(
            get_results_filename("figures/sigma_cmd.pdf"),
            category="Catalog",
        )
    conda:
        "../envs/figures.yml"
    log:
        get_log_filename("figures/sigma_cmd.log")
    shell:
        """
        python workflow/scripts/figures/sigma_cmd.py \\
            --input {input} \\
            --output {output} \\
            &> {log}
        """

rule figures_binary_fraction_cmd:
    input:
        get_results_filename("noise/apply.fits.gz")
    output:
        report(
            get_results_filename("figures/binary_fraction_cmd.pdf"),
            category="Catalog",
        )
    params:
        threshold=config["det_pval_thresh"]
    conda:
        "../envs/figures.yml"
    log:
        get_log_filename("figures/binary_fraction_cmd.log")
    shell:
        """
        python workflow/scripts/figures/binary_fraction_cmd.py \\
            --input {input} \\
            --output {output} \\
            --threshold {params.threshold} \\
            &> {log}
        """

rule figures_p_value_dist:
    input:
        get_results_filename("noise/apply.fits.gz")
    output:
        report(
            get_results_filename("figures/p_value_dist.pdf"),
            category="Catalog",
        )
    params:
        threshold=config["det_pval_thresh"]
    conda:
        "../envs/figures.yml"
    log:
        get_log_filename("figures/p_value_dist.log")
    shell:
        """
        python workflow/scripts/figures/p_value_dist.py \\
            --input {input} \\
            --output {output} \\
            --threshold {params.threshold} \\
            &> {log}
        """
