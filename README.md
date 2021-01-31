# one-datum

What can we infer about an orbit from the Gaia RV or astrometric jitter?

## Usage

This project is meant to be run containerized (although it'll probably also work
fine with a standard scientific Python installation). These instructions will
use [Singularity](https://sylabs.io), but this should also all work using
[Docker](https://www.docker.com).

To start, clone the project:

```bash
git clone https://github.com/dfm/one-datum.git
cd one-datum
```

### Downloading the data

To download all the necessary data files (using wget), run

```bash
scripts/download /path/to/data
```

where `/path/to/data` is the local path where you want to store the data files.

### Running the container

You can run the container using Singularity as follows (on a module system, you
might need to `module load singularity` first):

```bash
singularity_exec () {
  singularity exec --bind /path/to/data:/data docker://ghcr.io/dfm/one-datum:main $@;
}
singularity_exec python
```

where `/path/to/data` is the data path that you used above, and on the first
line we're defining a wrapper function that we'll continue using below. The
wrapper function isn't necessary, but it'll make the demos easier to parse. This
should drop you into a Python instance with a correctly configured environment.

### Exploratory analysis with Jupyterlab

This container comes with Jupyterlab installed and you can start a server in
this environment with:

```bash
singularity_exec jupyter lab --ip='*'
```

Since I'm normally running this on a remote machine, I would generally add a
specific port (`--port=8898`, for example) and then forward that port via SSH to
my development machine.

### Inferring per-transit RV uncertainty

This is done on a grid in BP-RP color and apparent G-magnitude. Use

```bash
singularity_exec scripts/infer-per-transit-uncert
```

to run with the default arguments or add the `--help` flag to see the available
parameters. This will save a file `rv_uncertainty_grid.fits` in your data
directory.
