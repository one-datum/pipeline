XMATCHES = [
    get_filename_for_dataset(f"xmatch/{xmatch}.fits.gz")
    for xmatch in [
        "sb9",
        "apogee-gold",
        "kepler",
    ]
]

rule xmatch_sb9:
    input:
        reference=get_results_filename("sb9.fits"),
        table=get_results_filename("inference/inferred.fits.gz")
    output:
        filename=get_results_filename("xmatch/sb9.fits.gz"),
        figure=report(
            get_results_filename("figures/sb9.pdf"),
            category="Crossmatch",
        )
    params:
        threshold=config["det_pval_thresh"]
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("xmatch/sb9.log")
    shell:
        """
        python workflow/scripts/xmatch.py \\
            --reference {input.reference} \\
            --table {input.table} \\
            --output {output.filename} \\
            --kcol 'sb9_k1' \\
            --name 'SB9' \\
            --figure {output.figure} \\
            --threshold {params.threshold} \\
            &> {log}
        """

rule xmatch_apogee_gold:
    input:
        reference=get_remote_filename("gold_sample.fits"),
        table=get_results_filename("inference/inferred.fits.gz")
    output:
        filename=get_results_filename("xmatch/apogee-gold.fits.gz"),
        figure=report(
            get_results_filename("figures/apogee-gold.pdf"),
            category="Crossmatch",
        )
    params:
        threshold=config["det_pval_thresh"]
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("xmatch/apogee-gold.log")
    shell:
        """
        python workflow/scripts/xmatch.py \\
            --reference {input.reference} \\
            --table {input.table} \\
            --output {output.filename} \\
            --source-id-col 'GAIAEDR3_SOURCE_ID' \\
            --kcol 'MAP_K' \\
            --kerrcol 'MAP_K_err' \\
            --name 'APOGEE' \\
            --figure {output.figure} \\
            --threshold {params.threshold} \\
            &> {log}
        """

rule xmatch_kepler:
    input:
        reference=get_remote_filename("kepler_edr3_1arcsec.fits"),
        table=get_results_filename("inference/inferred.fits.gz")
    output:
        filename=get_results_filename("xmatch/kepler.fits.gz")
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("xmatch/kepler.log")
    shell:
        """
        python workflow/scripts/xmatch.py \\
            --reference {input.reference} \\
            --table {input.table} \\
            --output {output.filename} \\
            &> {log}
        """
