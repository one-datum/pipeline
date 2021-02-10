FROM python:3.9
LABEL MAINTAINER "Dan Foreman-Mackey <foreman.mackey@gmail.com>"

# Install fonts for figures
RUN apt-get update \
 && apt-get install -y \
    fonts-liberation \
 && rm -rf /var/lib/apt/lists/*

COPY workflow/envs/requirements.txt .
RUN python -m pip install -U pip \
 && python -m pip install -r requirements.txt
