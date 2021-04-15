rule xmatch_sb9:
    input:
        reference=get_results_filename("sb9.fits"),
        table=get_results_filename("{dataset}/inference/inferred.fits.gz")
    output:
        filename=get_results_filename("{dataset}/xmatch/sb9.fits.gz"),
        figure=report(
            get_results_filename("{dataset}/xmatch/sb9.pdf"),
            category="Crossmatch",
        )
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/xmatch/sb9.log")
    shell:
        """
        python workflow/scripts/xmatch.py \\
            --reference {input.reference} \\
            --table {input.table} \\
            --output {output.filename} \\
            --figure {output.figure} \\
            &> {log}
        """
