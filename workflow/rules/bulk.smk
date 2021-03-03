# Bulk inference model

rule bulk_inference:
    input:
        "results/{dataset}/noise/estimated.fits.gz"
    output:
        "results/{dataset}/bulk/processed.fits.gz"
    conda:
        "../envs/environment.yml"
    log:
        stderr="results/logs/{dataset}/bulk/bulk-inference.log"
    shell:
        "python workflow/scripts/apply_noise_model.py --input {input} --output {output} &> {log}"
