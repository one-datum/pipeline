This directory includes various tools for automatically generating the figures
for the paper. These are designed to be run within a Docker container. To build
the image, navigate to this `paper/figures` directory and run

```bash
docker build -t one-datum-figures .
```

Then **in the root directory of the project** (not the figures directory), run:

```bash
docker run --rm --workdir /workspace -v $(pwd):/workspace one-datum-figures
```

to run the code that generates the figures.
