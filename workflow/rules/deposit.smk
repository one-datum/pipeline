rule archive_figures:
    input:
        directory=get_results_filename("{dataset}/figures"),
        figures=FIGURES,
        xmatches=XMATCHES
    output:
        get_results_filename("{dataset}/figures.tar.gz")
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/figures.tar.gz.log")
    shell:
        """
        tar czvfC {output} `dirname "{input.directory}"` `basename "{input.directory}"` &> {log}
        """
