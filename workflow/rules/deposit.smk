ARCHIVE_DEPS = {
    "archive": [
        get_results_filename("archive/inference/inferred.fits.gz"),
        get_results_filename("archive/simulations/inferred.fits.gz"),
        get_results_filename("archive/noise/process.fits"),
    ],
    "figures": FIGURES + XMATCHES,
    "xmatch": XMATCHES,
}

rule archive_directory:
    output:
        directory(get_results_filename("{archive}"))
    wildcard_constraints:
        archive="[a-zA-Z]+"
    conda:
        "../envs/baseline.yml"
    log:
        get_log_filename("{archive}.log")
    shell:
        "mkdir -p {output} &> {log}"

rule archive_copy:
    input:
        filename=get_results_filename("{filename}"),
        directory=get_results_filename("archive")
    output:
        get_results_filename("archive/{filename}")
    conda:
        "../envs/baseline.yml"
    log:
        get_log_filename("archive/{filename}.log")
    shell:
        "cp -r {input.filename} {output} &> {log}"

rule archive_zip:
    input:
        directory=get_results_filename("{archive}"),
        files=lambda wildcards: ARCHIVE_DEPS[wildcards["archive"]]
    output:
        get_results_filename("{archive}.zip")
    wildcard_constraints:
        archive="[a-zA-Z]+"
    conda:
        "../envs/baseline.yml"
    log:
        get_log_filename("{archive}.zip.log")
    shell:
        """
        cd `dirname "{input.directory}"`; zip -r {output} `basename "{input.directory}"` -x '*.snakemake*' &> {log}
        """

rule deposit_results:
    input:
        get_results_filename("{target}.zip")
    output:
        get_results_filename("{target}.zenodo")
    params:
        creds=config["zenodo"],
        metadata="workflow/metadata/{target}.yaml"
    conda:
        "../envs/baseline.yml"
    log:
        get_log_filename("{target}.zenodo.log")
    shell:
        """
        python workflow/scripts/upload.py \\
            --input {input} \\
            --output {output} \\
            --metadata {params.metadata} \\
            --creds {params.creds} \\
            &> {log}
        """
