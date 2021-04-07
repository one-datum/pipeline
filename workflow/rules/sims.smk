rule simulate_catalog:
    input:
        config["base_table_filename"]
    output:
        "results/{dataset}/sims/simulated.fits.gz"
    params:
        config=config["sims_config_file"],
    conda:
        "../envs/environment.yml"
    log:
        "results/logs/{dataset}/sims/simulate-catalog.log"
    shell:
        """
        python workflow/scripts/simulate_catalog.py \\
        --input {input} \\
        --output {output} \\
        --config {params.config} \\
        &> {log}
        """
