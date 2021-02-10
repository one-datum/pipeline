FROM continuumio/miniconda3:latest
LABEL MAINTAINER "Dan Foreman-Mackey <foreman.mackey@gmail.com>"

# Install fonts for figures
RUN apt-get update \
 && apt-get install -y \
    fonts-liberation \
 && rm -rf /var/lib/apt/lists/*

# Install conda-merge for building the environment
RUN python -m pip install -U pip \
 && python -m pip install conda-merge

# Set up the conda environment
COPY workflow/envs /envs
RUN conda-merge /envs/* > /envs/environment.yml \
 && cat /envs/environment.yml \
 && conda install -c conda-forge mamba \
 && mamba env create --name one-datum --file /envs/environment.yml
