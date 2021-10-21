rule upload:
    input:
        get_results_filename("{catchall}/{target}")
    output:
        get_results_filename("{catchall}/{target}.zenodo")
    params:
        creds=config["zenodo"],
        metadata="workflow/metadata/{target}.yaml"
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{catchall}/{target}.zenodo.log")
    shell:
        """
        python workflow/scripts/upload.py \\
            --input {input} \\
            --output {output} \\
            --metadata {params.metadata} \\
            --creds {params.creds} \\
            &> {log}
        """
