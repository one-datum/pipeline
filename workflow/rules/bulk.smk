# Bulk inference model

rule bulk_inference:
    input:
        get_simulated_or_real_catalog
    output:
        "results/{dataset}/bulk/processed.fits.gz"
    conda:
        "../envs/environment.yml"
    log:
        "results/logs/{dataset}/bulk/bulk-inference.log"
    shell:
        "python workflow/scripts/bulk_inference.py --input {input} --output {output} &> {log}"
