FROM python:3.11.6-slim-bullseye

RUN apt-get update
RUN apt-get install -y default-jre
RUN apt-get install -y gcc

WORKDIR /app/src

RUN pip install --no-cache-dir --upgrade poetry==1.5.0

COPY ./src/pyproject.toml ./src/poetry.lock* /app/src/

RUN pip install lightfm==1.17 --no-use-pep517
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

COPY ./src /app/src
COPY ./README.md /app/README.md

ARG DATA_TAG=v4

RUN mkdir /data
COPY ./data/${DATA_TAG}/tracks.jsonl /data/${DATA_TAG}/tracks.jsonl
COPY ./data/${DATA_TAG}/artists.jsonl /data/${DATA_TAG}/artists.jsonl


RUN mkdir /predictions

WORKDIR /app

EXPOSE 8081

ENTRYPOINT ["python3", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8081"]
