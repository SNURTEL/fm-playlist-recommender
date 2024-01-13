from datetime import datetime

from fastapi import FastAPI, responses, HTTPException

from src.logger import get_logger
from src.schemas import PredictRequestSchema, PredictionListEntrySchema
from src.util import dump_results, PredictionHistoryRecord, choose_model, setup_recommender

DATA_VERSION = "v4"

app = FastAPI()

logger = get_logger()

recommender = setup_recommender(data_version="v4")


@app.get("/", response_class=responses.PlainTextResponse)
def root():
    """
    Get project description
    """
    with open("README.md") as fp:
        return fp.read()


@app.post("/predict")
def predict_ab(prs: PredictRequestSchema) -> list[PredictionListEntrySchema]:
    """
    Make recommendations using either base or advanced model, depending on supplied user ID.
    """
    picked_model = choose_model(prs.user_id)

    if picked_model == 'base':
        res, t = recommender.generate_recommendation_base(prs.users, prs.playlist_length)
        logger.debug(f"Model [BASE]/advanced\t{prs.playlist_length} songs")
    elif picked_model == 'advanced':
        res, t = recommender.generate_recommendation_advanced(prs.users, prs.playlist_length)
        logger.debug(f"Model base/[ADVANCED]\t{prs.playlist_length} songs")
    else:
        raise HTTPException(500)

    json: PredictionHistoryRecord = {
        'user_id': prs.user_id,
        'other_users': prs.other_users,
        'playlist': res.index.values.tolist(),
        'timestamp': datetime.now(),
        'elapsed_time': t,
        'model': picked_model,
        'is_ab': True
    }

    dump_results(json)

    return [
        PredictionListEntrySchema(id=track_id, name=name, artist=artist) for track_id, (name, artist) in res.iterrows()
    ]


@app.post("/predict/base")
def predict_base(prs: PredictRequestSchema) -> list[PredictionListEntrySchema]:
    """
    Make recommendations using the base model.
    """
    res, t = recommender.generate_recommendation_base(prs.users, prs.playlist_length)
    logger.debug(f"Model BASE\t{prs.playlist_length} songs")

    json: PredictionHistoryRecord = {
        'user_id': prs.user_id,
        'other_users': prs.other_users,
        'playlist': res.index.values.tolist(),
        'timestamp': datetime.now(),
        'elapsed_time': t,
        'model': 'base',
        'is_ab': False
    }

    dump_results(json)

    return [
        PredictionListEntrySchema(id=track_id, name=name, artist=artist) for track_id, (name, artist) in res.iterrows()
    ]


@app.post("/predict/advanced")
def predict_advanced(prs: PredictRequestSchema) -> list[PredictionListEntrySchema]:
    """
    Make recommendations using the advanced model.
    """
    res, t = recommender.generate_recommendation_advanced(prs.users, prs.playlist_length)
    logger.debug(f"Model ADVANCED\t{prs.playlist_length} songs")

    json: PredictionHistoryRecord = {
        'user_id': prs.user_id,
        'other_users': prs.other_users,
        'playlist': res.index.values.tolist(),
        'timestamp': datetime.now(),
        'elapsed_time': t,
        'model': 'base',
        'is_ab': False
    }

    dump_results(json)

    return [
        PredictionListEntrySchema(id=track_id, name=name, artist=artist) for track_id, (name, artist) in res.iterrows()
    ]
