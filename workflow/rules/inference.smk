# Bulk inference model

rule inference:
    input:
        get_results_filename("{dataset}/noise/applied.fits.gz")
    output:
        get_results_filename("{dataset}/inference/inferred.fits.gz")
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/inference.log")
    shell:
        "python workflow/scripts/inference.py --input {input} --output {output} &> {log}"
