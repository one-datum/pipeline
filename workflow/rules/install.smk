# A rule to install the working copy of this package

rule install_local:
    input:
        "workflow/envs/environment.yml"
    output:
        touch("results/installed.done")
    conda:
        "../envs/environment.yml"
    log:
        "results/logs/install-local.log"
    shell:
        "echo \"nothing\"""
        # "python -m pip install -e . &> {log}"
