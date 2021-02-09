# one-datum

What can we infer about an orbit from the Gaia RV or astrometric jitter?

## Usage

This project includes a pipeline component and a user-facing library.
For now, this README just covers the pipeline usage which is designed to be run using [snakemake](https://snakemake.readthedocs.io).

### Environment setup

To get started, you should create a simple conda environment and install _snakemake_ as follows:

```bash
conda install -c conda-forge mamba
mamba create -n one-datum -c conda-forge bioconda::snakemake-minimal
conda activate one-datum
```

Then you can clone this repository to get the pipeline workflow.
Note that the results files will be stored in the `results` and `resources` subdirectories so make sure that you run this command on a drive with a fairly large quota (about TBD will be required):

```bash
git clone https://github.com/dfm/one-datum.git
cd one-datum
```

### Inferring per-transit RV uncertainty

This is done on a grid in BP-RP color and apparent G-magnitude. Use

```bash
snakemake infer_noise --cores all --use-conda --conda-frontend mamba
```

to run with the default arguments.
This will save a file `rv_uncertainty_grid.fits` in your data directory.
