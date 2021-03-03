import yaml
from snakemake.utils import validate

validate(config, "../schemas/config.schema.yaml")

with open(config["noise_config_file"], "r") as f:
    noise_config = yaml.load(f.read(), Loader=yaml.FullLoader)
validate(noise_config, "../schemas/noise.schema.yaml")


def get_final_output():
    return ["results/data/{0}/processed.fits.gz".format(config["dataset_name"])]


def get_bulk_filenames():
    filenames = []
    if config["install_noise_model"]:
        filenames.append("src/one_datum/data/{0}-noise-model.fits".format(config["dataset_name"]))
    filenames.append(config["base_table_filename"])
    return filenames
