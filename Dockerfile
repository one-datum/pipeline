FROM python:3.8

WORKDIR /code

COPY requirements.txt .
RUN python -m pip install -U pip \
 && python -m pip install -r requirements.txt

CMD [ "python", "--version" ]
