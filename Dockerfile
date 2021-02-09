FROM python:3.8
WORKDIR /code

RUN apt-get update \
 && apt-get install -y \
    fonts-liberation \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install -U pip \
 && python -m pip install -r requirements.txt

COPY . /build
RUN python -m pip install /build

CMD [ "python", "--version" ]
