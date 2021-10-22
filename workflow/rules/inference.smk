rule inference:
    input:
        get_results_filename("noise/apply.fits.gz")
    output:
        get_results_filename("inference/inferred.fits.gz")
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("inference.log")
    shell:
        "python workflow/scripts/inference.py --input {input} --output {output} &> {log}"
