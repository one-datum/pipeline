import json

# # This rule is no longer used since we download the query from Zenodo 
# rule base:
#     output:
#         get_results_filename("base.fits.gz")
#     params:
#         gaia=json.dumps(config["gaia"])
#     conda:
#         "../envs/baseline.yml"
#     log:
#         get_log_filename("base.log")
#     shell:
#         """
#         python workflow/scripts/query.py \\
#             --output {output} \\
#             --gaia-creds '{params.gaia}' \\
#             &> {log}
#         """

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
        "../envs/baseline.yml"
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
