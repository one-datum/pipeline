# one-datum
What can we infer about an orbit from the RV or astrometric jitter?


## Usage

This project is meant to be run containerized (although it'll probably also work fine with a standard scientific Python installation).
These instructions will use [Singularity](https://sylabs.io), but this should also all work using [Docker](https://www.docker.com).

To start, clone the project:

```bash
git clone https://github.com/dfm/one-datum.git
cd one-datum
```

### Downloading the data

To download all the necessary data files (using wget), run

```bash
./download.sh /path/to/data
```

where `/path/to/data` is the local path where you want to store the data files.

### Running the container

You can run the container using Singularity as follows (on a module system, you might need to `module load singularity` first):

```bash
singularity exec --bind /path/to/data:/data docker://ghcr.io/dfm/one-datum:main python
```

where `/path/to/data` is the data path that you used above.
This should drop you into a Python instance with a correctly configured environment.
