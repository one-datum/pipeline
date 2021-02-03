FROM python:3.8

RUN apt-get update \
 && apt-get install -y \
    fonts-liberation \
 && rm -rf /var/lib/apt/lists/*

COPY paper/figures/requirements.txt requirements.txt
RUN python -m pip install -U pip \
 && python -m pip install -r requirements.txt

COPY . /build
RUN python -m pip install /build

ENTRYPOINT paper/figures/make-figures.sh
