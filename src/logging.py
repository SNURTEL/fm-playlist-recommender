import logging
from logging.config import dictConfig

from pydantic import BaseModel

LOG_TAG = "RECOMMENDER_SERVICE"


class LogConfig(BaseModel):
    LOGGER_NAME: str = LOG_TAG
    LOG_FORMAT: str = "%(levelprefix)s | %(asctime)s | %(message)s"
    LOG_LEVEL: str = "DEBUG"

    version: int = 1
    disable_existing_loggers: bool = False
    formatters: dict = {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    }
    handlers: dict = {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    }
    loggers: dict = {
        LOGGER_NAME: {"handlers": ["default"], "level": LOG_LEVEL},
    }


dictConfig(LogConfig().model_dump())


def get_logger() -> logging.Logger:
    return logging.getLogger(LOG_TAG)