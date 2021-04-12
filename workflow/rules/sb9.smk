# Rules for downloading and crossmatching to the SB9 catalog

localrules: sb9_data

rule sb9_data:
    input:
        "resources/data/SB9public.tar.gz"
    output:
        directory(get_remote_filename("SB9public"))
    log:
        get_log_filename("sb9/data.log")
    shell:
        "mkdir -p {output};tar -xzvf {input} -C {output} 2> {log}"

rule sb9_xmatch:
    input:
        get_remote_filename("SB9public")
    output:
        get_remote_filename("sb9-gaia-xmatch.fits")
    conda:
        "../envs/environment.yml"
    log:
        notebook=get_log_filename("notebooks/sb9-gaia-xmatch.ipynb")
    notebook:
        "../notebooks/sb9-gaia-xmatch.py.ipynb"
