rule archive_directory:
    output:
        directory(get_results_filename("archive"))
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("archive.log")
    shell:
        "mkdir -p {output} &> {log}"

rule copy_result_to_archive:
    input:
        filename=get_results_filename("{filename}"),
        directory=get_results_filename("archive")
    output:
        get_results_filename("archive/{filename}")
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("archive/{filename}.log")
    shell:
        "cp -r {input.filename} {output} &> {log}"

rule archive_results:
    input:
        get_results_filename("archive/inference/inferred.fits.gz"),
        get_results_filename("archive/simulations/inferred.fits.gz"),
        get_results_filename("archive/noise/process.fits"),
        directory=get_results_filename("archive")
    output:
        get_results_filename("archive.tar.gz")
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("archive.tar.gz.log")
    shell:
        """
        tar czvfC {output} `dirname "{input.directory}"` `basename "{input.directory}"` &> {log}
        """

rule figures_directory:
    output:
        directory(get_results_filename("figures"))
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("figures.log")
    shell:
        "mkdir -p {output} &> {log}"

rule archive_figures:
    input:
        directory=get_results_filename("figures"),
        figures=FIGURES,
        xmatches=XMATCHES
    output:
        get_results_filename("figures.tar.gz")
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("figures.tar.gz.log")
    shell:
        """
        tar czvfC {output} `dirname "{input.directory}"` `basename "{input.directory}"` &> {log}
        """

rule xmatch_directory:
    output:
        directory(get_results_filename("xmatch"))
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("xmatch.log")
    shell:
        "mkdir -p {output} &> {log}"

rule archive_xmatch:
    input:
        directory=get_results_filename("xmatch"),
        xmatches=XMATCHES
    output:
        get_results_filename("xmatch.tar.gz")
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("xmatch.tar.gz.log")
    shell:
        """
        tar czvfC {output} `dirname "{input.directory}"` `basename "{input.directory}"` &> {log}
        """

rule deposit_results:
    input:
        get_results_filename("{target}.tar.gz")
    output:
        get_results_filename("{target}.zenodo")
    params:
        creds=config["zenodo"],
        metadata="workflow/metadata/{target}.yaml"
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("{target}.zenodo.log")
    shell:
        """
        python workflow/scripts/upload.py \\
            --input {input} \\
            --output {output} \\
            --metadata {params.metadata} \\
            --creds {params.creds} \\
            --sandbox \\
            &> {log}
        """
