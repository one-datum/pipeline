# Bulk inference model

rule bulk_inference:
    input:
        get_bulk_filenames()
    output:
        "results/data/{dataset}/processed.fits.gz"
    conda:
        "../envs/environment.yml"
    log:
        "results/logs/bulk/{dataset}/bulk-inference.log"
    script:
        "../scripts/apply_noise_model.py"
