# one-datum

What can we infer about an orbit from the Gaia RV jitter?

## Usage

This project includes a pipeline component and a user-facing library.
For now, this README just covers the pipeline usage which is designed to be run using [snakemake](https://snakemake.readthedocs.io).

### Environment setup

To get started, you should create a simple conda environment and install _snakemake_ as follows:

```bash
conda install -c conda-forge mamba
mamba create -n one-datum -c conda-forge bioconda::snakemake-minimal networkx pygraphviz
conda activate one-datum
```

Then you can clone this repository to get the pipeline workflow:

```bash
git clone https://github.com/dfm/one-datum.git
cd one-datum
```


### Configuration

You can configure the workflow using a custom `config.yaml` and/or a [Snakemake profile](https://snakemake.readthedocs.io/en/stable/executing/cli.html#profiles).
For example, I normally use a profile with the following settings:

```yaml
# $HOME/.config/snakemake/one-datum/config.yaml
cores: all
use-conda: true
conda-frontend: mamba
```

and then I would execute Snakemake as follows:

```bash
snakemake --profile=one-datum
```

where `one-datum` is the name of the directory where the profile configuration file is saved.

Take a look at the [default config files](https://github.com/dfm/one-datum/tree/main/config) for all the options, but you might want to explicitly set `remote_basedir` and `results_basedir` parameters so that large files don't get written to your working directory.


### Running the pipeline

The following command should run the full pipeline and produce a catalog estimated binary parameters for all Gaia EDR3 radial velocity sources:

```bash
snakemake --profile=one-datum
```

Make sure that you run this on a beefy machine.


### Running simulations & estimating completeness

You can also run some simulations for characterizing the pipeline and computing the completeness.
To do that, run:

```bash
snakemake --profile=one-datum completeness
```

The settings for the simulations can all be found in the `config/simulations.yaml` configuration file.
