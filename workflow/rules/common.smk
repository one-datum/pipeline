from snakemake.utils import validate

validate(config, "../schemas/config.schema.yaml")

def get_final_output():
    return ["results/data/{0}/processed.fits.gz".format(config["dataset_name"])]


def get_bulk_filenames():
    filenames = []
    if config["noise"]["install_noise_model"]:
        filenames.append("src/one_datum/data/{0}-noise-model.fits".format(config["dataset_name"]))
    filenames.append(config["noise"]["base_table_filename"])
    return filenames
