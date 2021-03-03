# A rule to install the working copy of this package

rule install_local:
    output:
        touch("results/installed.done")
    conda:
        "../envs/environment.yml"
    log:
        "results/logs/install-local.log"
    shell:
        "python -m pip install -e . &> {log}"
