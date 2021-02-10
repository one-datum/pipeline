FROM ubuntu:20.04
LABEL MAINTAINER "Dan Foreman-Mackey <foreman.mackey@gmail.com>"

# Install system dependencies
RUN apt-get update \
 && apt-get install -y \
    fonts-liberation \
    git \
    wget \
 && rm -rf /var/lib/apt/lists/*

# Install micromamba
ENV MAMBA_ROOT_PREFIX $HOME/micromamba
RUN wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest \
  | tar -xvj bin/micromamba \
 && micromamba shell init -s bash \
 && echo "micromamba activate $MAMBA_ROOT_PREFIX" >> $HOME/.bashrc \
 && micromamba install mamba -c conda-forge --prefix $MAMBA_ROOT_PREFIX
ENV PATH $MAMBA_ROOT_PREFIX/bin:${PATH}

# Set up the environment and install pipeline dependencies
COPY workflow/envs/environment.yml environment.yml
RUN $MAMBA_ROOT_PREFIX/bin/mamba env update --file environment.yml --prefix $MAMBA_ROOT_PREFIX \
 && rm environment.yml

# # Installing pipeline dependencies
# COPY workflow/envs/requirements.txt .
# RUN python -m pip install -U pip \
#  && python -m pip install -r requirements.txt
