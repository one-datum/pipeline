rule install:
    input:
        "src/one_datum"
    output:
        touch(get_results_filename("install.done"))
    log:
        get_log_filename("install.log")
    conda:
        "../envs/environment.yml"
    shell:
        "python -m pip install -e . &> {log}"
