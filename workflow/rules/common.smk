import yaml
from snakemake.utils import validate

validate(config, "../schemas/config.schema.yaml")

with open(config["noise_config_file"], "r") as f:
    noise_config = yaml.load(f.read(), Loader=yaml.FullLoader)
validate(noise_config, "../schemas/noise.schema.yaml")


def get_final_output():
    return ["results/{0}/bulk/processed.fits.gz".format(config["dataset_name"])]


def get_simulated_or_real_catalog(wildcards):
    if config["simulate_catalog"]:
        # with open(config["sims_config_file"], "r") as f:
        #     sims_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        # validate(sims_config, "../schemas/sims.schema.yaml")
        return [f"results/{wildcards.dataset}/sims/simulated.fits.gz"]
    return [f"results/{wildcards.dataset}/noise/estimated.fits.gz"]
