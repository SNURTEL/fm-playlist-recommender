import json
import pickle
import sys
from typing import TypedDict
from datetime import timedelta, datetime

import src.model as src_model
from src.recommender import Recommender
from src.spark.config import views
from src.spark.create_session import create_session


def setup_recommender(data_version: str) -> Recommender:
    sys.modules['model'] = src_model

    VIEWS = views(data_version, filter_items=['tracks', 'artists'])
    spark = create_session()

    for view, file in VIEWS.items():
        df = spark.read.json(file)
        df.createOrReplaceTempView(view)

    _tracks_with_artists = spark.sql(
        f"SELECT DISTINCT t.id as track_id, t.name as track_name, a.id as artist_id, a.name as artist_name "
        f"FROM tracks t join artists a on t.id_artist == a.id").toPandas()

    with open("/app/src/serialized/base_model.bin", mode='rb') as fp:
        base_model = pickle.load(fp)

    with open("/app/src/serialized/advanced_model.bin", mode='rb') as fp:
        advanced_model = pickle.load(fp)

    del sys.modules['model']

    return Recommender(
        base_model=base_model,
        advanced_model=advanced_model,
        tracks_with_artists_df=_tracks_with_artists
    )


def choose_model(user_id: int) -> Recommender.model_t:
    return 'base' if hash(user_id) % 2 == 0 else 'advanced'


class PredictionHistoryRecord(TypedDict):
    user_id: int
    other_users: list[int]
    playlist: list[str]
    timestamp: datetime
    elapsed_time: timedelta
    model: Recommender.model_t
    is_ab: bool


def dump_results(res: PredictionHistoryRecord):
    with open("/predictions/predictions.jsonl", "a") as fp:
        json.dump(res, fp, default=str)
        fp.write('\n')
