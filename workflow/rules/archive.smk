import json

rule base:
    output:
        get_results_filename("{dataset}/base.fits.gz")
    params:
        gaia=json.dumps(config["gaia"])
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/base.log")
    shell:
        """
        python workflow/scripts/query.py \\
            --output {output} \\
            --gaia-creds '{params.gaia}' \\
            &> {log}
        """

rule sb9:
    output:
        filename=get_results_filename("sb9.fits"),
        figure=report(
            get_results_filename("sb9.pdf"),
            category="SB9",
        )
    params:
        gaia=json.dumps(config["gaia"])
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("sb9.log")
    shell:
        """
        python workflow/scripts/sb9.py \\
            --output {output.filename} \\
            --figure {output.figure} \\
            --gaia-creds '{params.gaia}' \\
            &> {log}
        """