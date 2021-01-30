FROM python:3.8

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt .
RUN python -m pip install -U pip \
 && python -m pip install -r requirements.txt

CMD [ "python", "--version" ]
