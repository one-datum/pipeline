rule archive_figures:
    input:
        [get_results_filename("{dataset}/figures")] + FIGURES + XMATCHES
    output:
        get_results_filename("{dataset}/figures.tar.gz")
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/figures.tar.gz.log")
    shell:
        """
        tar czvfC {output} `dirname "{input[0]}"` `basename "{input[0]}"` &> {log}
        """
