FROM python:3.11.6-slim-bullseye

ARG DATA_TAG=v3

RUN apt-get update
RUN apt-get install -y default-jre
RUN apt-get install -y gcc

WORKDIR /app/src

RUN pip install --no-cache-dir --upgrade poetry==1.5.0

COPY ./src/pyproject.toml ./src/poetry.lock* /app/src/

RUN pip install lightfm==1.17 --no-use-pep517
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

COPY ./src /app/src

RUN mkdir /data
VOLUME ["/data"]
COPY ./data/${DATA_TAG} /data/${DATA_TAG}


RUN mkdir /predictions
VOLUME ["/predictions"]

# copy individual files
#COPY ./data/${DATA_TAG}/tracks.jsonl /data/${DATA_TAG}/tracks.jsonl

WORKDIR /app

EXPOSE 8081

ENTRYPOINT ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8081"]
