This directory includes various tools for automatically generating the figures
for the paper. These are designed to be run within a Docker container.

If you don't want to build the figures yourself, download an archive of the most
recently generated figures and unpack it in the root directory of this repo:

```bash
wget https://github.com/dfm/one-datum/archive/main-pdf.tar.gz
tar --strip-components=1 -xvf main-pdf.tar.gz
```

## Generating the figures yourself

To build the image, navigate to this `paper/figures` directory and run

```bash
docker build -t one-datum-figures .
```

Then **in the root directory of the project** (not the figures directory), run:

```bash
docker run --rm --workdir /workspace -v $(pwd):/workspace one-datum-figures
```

to run the code that generates the figures.
