rule inference:
    input:
        catalog=get_results_filename("{dataset}/noise/pval.fits.gz"),
        calib=get_results_filename("{dataset}/noise/calibrate.txt")
    output:
        get_results_filename("{dataset}/inference/inferred.fits.gz")
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/inference.log")
    shell:
        """
        python workflow/scripts/inference.py \\
            --catalog {input.catalog} \\
            --calib {input.calib} \\
            --output {output} \\
            &> {log}
        """
