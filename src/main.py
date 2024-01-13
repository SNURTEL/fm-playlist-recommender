from datetime import datetime

from fastapi import FastAPI, responses, HTTPException

from src.logging import get_logger
from src.schemas import PredictRequestSchema, PredictionListEntrySchema
from src.util import generate_recommendation_base, dump_results, PredictionHistoryRecord, \
    generate_recommendation_advanced, choose_model

app = FastAPI()

logger = get_logger()

@app.get("/", response_class=responses.PlainTextResponse)
def root():
    return """IUM Task 3 Variant 2
By Tomasz Owienko and Anna Schäfer

Example:
http://0.0.0.0:8081/predict/1?playlist_length=10&other_users=2,3,4
For base model:
http://0.0.0.0:8081/predict/base_model/1?playlist_length=10&other_users=2,3,4
For advanced model:
http://0.0.0.0:8081/predict/advanced_model/1?playlist_length=10&other_users=2,3,4

You can also use swagger under:
http://0.0.0.0:8081/docs#/

Results are saved in predictions.jsonl
"""


@app.post("/predict")
def predict_ab(prs: PredictRequestSchema) -> list[PredictionListEntrySchema]:
    if choose_model(prs.user_id) == 'base':
        res, t = generate_recommendation_base(prs.users, prs.playlist_length)
        logger.debug(f"Model [BASE]/advanced\t{prs.playlist_length} songs")
    elif choose_model(prs.user_id) == 'advanced':
        res, t = generate_recommendation_advanced(prs.users, prs.playlist_length)
        logger.debug(f"Model base/[ADVANCED]\t{prs.playlist_length} songs")
    else:
        raise HTTPException(500)

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


@app.post("/predict/base")
def predict_base(prs: PredictRequestSchema) -> list[PredictionListEntrySchema]:
    res, t = generate_recommendation_base(prs.users, prs.playlist_length)
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
    res, t = generate_recommendation_advanced(prs.users, prs.playlist_length)
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
