# Rules for downloading and crossmatching to the SB9 catalog

localrules: sb9_data

rule sb9_data:
    input:
        "resources/data/SB9public.tar.gz"
    output:
        directory("resources/data/SB9public")
    log:
        "results/logs/sb9/data.log"
    shell:
        "mkdir -p {output};tar -xzvf {input} -C {output} 2> {log}"

rule sb9_xmatch:
    input:
        "resources/data/SB9public"
    output:
        "resources/data/sb9-gaia-xmatch.fits"
    conda:
        "../envs/environment.yml"
    log:
        notebook="results/logs/notebooks/sb9-gaia-xmatch.ipynb"
    notebook:
        "../notebooks/sb9-gaia-xmatch.py.ipynb"
