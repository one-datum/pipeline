# Bulk inference model

rule bulk_inference:
    input:
        get_bulk_filenames()
    output:
        "results/{dataset}/bulk/processed.fits.gz"
    conda:
        "../envs/environment.yml"
    log:
        "results/logs/{dataset}/bulk/bulk-inference.log"
    script:
        "../scripts/apply_noise_model.py"
