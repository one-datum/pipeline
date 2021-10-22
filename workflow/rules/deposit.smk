rule upload:
    input:
        get_results_filename("{catchall}/{target}")
    output:
        get_results_filename("{catchall}/{target}.zenodo")
    params:
        creds=config["zenodo"],
        metadata="workflow/metadata/{target}.yaml"
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{catchall}/{target}.zenodo.log")
    shell:
        """
        python workflow/scripts/upload.py \\
            --input {input} \\
            --output {output} \\
            --metadata {params.metadata} \\
            --creds {params.creds} \\
            &> {log}
        """

rule archive_directory:
    output:
        directory(get_results_filename("{dataset}/archive"))
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/archive.log")
    shell:
        "mkdir -p {output} &> {log}"

rule copy_result_to_archive:
    input:
        get_results_filename("{dataset}/{filename}")
    output:
        get_results_filename("{dataset}/archive/{filename}")
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/archive/{filename}.log")
    shell:
        "cp -r {input} {output} &> {log}"

rule archive_results:
    input:
        get_results_filename("{dataset}/archive/inference/inferred.fits.gz"),
        get_results_filename("{dataset}/archive/simulations/inferred.fits.gz"),
        get_results_filename("{dataset}/archive/noise/process.fits"),
        directory=get_results_filename("{dataset}/archive")
    output:
        get_results_filename("{dataset}/archive.tar.gz")
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/archive.tar.gz.log")
    shell:
        """
        tar czvfC {output} `dirname "{input.directory}"` `basename "{input.directory}"` &> {log}
        """

rule figures_directory:
    output:
        directory(get_results_filename("{dataset}/figures"))
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/figures.log")
    shell:
        "mkdir -p {output} &> {log}"

rule archive_figures:
    input:
        directory=get_results_filename("{dataset}/figures"),
        figures=FIGURES,
        xmatches=XMATCHES
    output:
        get_results_filename("{dataset}/figures.tar.gz")
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{dataset}/figures.tar.gz.log")
    shell:
        """
        tar czvfC {output} `dirname "{input.directory}"` `basename "{input.directory}"` &> {log}
        """
