# Bulk inference model

rule bulk_inference:
    input:
        get_results_filename("{dataset}/noise/estimated.fits.gz")
    output:
        get_results_filename("{dataset}/bulk/processed.fits.gz")
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/bulk/bulk-inference.log")
    shell:
        "python workflow/scripts/bulk_inference.py --input {input} --output {output} &> {log}"
