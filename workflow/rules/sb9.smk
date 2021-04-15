import json

localrules: sb9_data
ruleorder: sb9_data > get_data

# rule sb9_data:
#     input:
#         get_remote_filename("SB9public.tar.gz")
#     output:
#         directory(get_remote_filename("SB9public"))
#     conda:
#         "../envs/environment.yml"
#     log:
#         get_log_filename("sb9/data.log")
#     shell:
#         "mkdir -p {output};tar -xzvf {input} -C {output} 2> {log}"

rule sb9_data:
    input:
        get_remote_filename("sb9/{filename}.dat.gz")
    output:
        get_remote_filename("sb9/{filename}.dat")
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("sb9/{filename}.dat.log")
    shell:
        "gunzip -c {input} > {output} 2> {log}"


rule sb9_xmatch:
    input:
        readme=get_remote_filename("sb9/ReadMe"),
        main=get_remote_filename("sb9/main.dat"),
        orbits=get_remote_filename("sb9/orbits.dat")
        # get_remote_filename("SB9public")
    output:
        get_remote_filename("sb9-gaia-xmatch.fits")
    params:
        gaia=json.dumps(config["gaia"])
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("sb9/xmatch.log")
    shell:
        """
        python workflow/scripts/xmatch/sb9.py \\
            --readme {input.readme} \\
            --main {input.main} \\
            --orbits {input.orbits} \\
            --output {output} \\
            --gaia-creds '{params.gaia}' \\
            &> {log}
        """
