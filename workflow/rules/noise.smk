with open(config["noise_config_file"], "r") as f:
    noise_config = yaml.load(f.read(), Loader=yaml.FullLoader)


rule noise_infer:
    input:
        get_results_filename(config["base_table_filename"])
    output:
        get_results_filename("noise/infer-{n}.fits")
    params:
        config=config["noise_config_file"],
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("noise/infer-{n}.log")
    shell:
        """
        python workflow/scripts/noise/infer.py \\
            --input {input} \\
            --output {output} \\
            --config {params.config} \\
            --mag-bin {wildcards.n} \\
            &> {log}
        """

rule noise_combine:
    input:
        expand(
            get_results_filename("noise/infer-{n}.fits"),
            n=range(noise_config["num_mag"]),
            dataset="{dataset}",
        )
    output:
        get_results_filename("noise/combine.fits")
    params:
        config=config["noise_config_file"],
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("noise/combine.log")
    shell:
        """
        python workflow/scripts/noise/combine.py \\
            --output {output} \\
            --config {params.config} \\
            {input} \\
            &> {log}
        """

rule noise_postprocess:
    input:
        get_results_filename("noise/combine.fits")
    output:
        get_results_filename("noise/process.fits")
    params:
        color_smooth=config["noise"]["color_smoothing_scale"],
        mag_smooth=config["noise"]["mag_smoothing_scale"]
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("noise/process.log")
    shell:
        """
        python workflow/scripts/noise/process.py \\
            --input {input} --output {output} \\
            --color-smooth {params.color_smooth} \\
            --mag-smooth {params.mag_smooth} \\
            &> {log}
        """

rule noise_apply:
    input:
        noise_model=get_results_filename("noise/process.fits"),
        base_table=get_results_filename(config["base_table_filename"])
    output:
        get_results_filename("noise/apply.fits.gz")
    conda:
        "../envs/environment.yml"
    log:
        get_log_filename("noise/apply.log")
    shell:
        """
        python workflow/scripts/noise/apply.py \\
            --noise-model {input.noise_model} \\
            --input {input.base_table} \\
            --output {output} \\
            &> {log}
        """
