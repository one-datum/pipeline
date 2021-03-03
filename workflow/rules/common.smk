import yaml
from snakemake.utils import validate

validate(config, "../schemas/config.schema.yaml")

with open(config["noise_config_file"], "r") as f:
    noise_config = yaml.load(f.read(), Loader=yaml.FullLoader)
validate(noise_config, "../schemas/noise.schema.yaml")


def get_final_output():
    return ["results/{0}/bulk/processed.fits.gz".format(config["dataset_name"])]
