import json
import time
from typing import Literal, TypedDict
from datetime import timedelta, datetime

import pandas as pd
from scipy import stats

from src.model.base_model import BaseModel
from src.spark.config import views
from src.spark.create_session import create_session

model_t = Literal['base', 'advanced']

VIEWS = views("v3")
spark = create_session()

for view, file in VIEWS.items():
    df = spark.read.json(file)
    df.createOrReplaceTempView(view)

_tracks = spark.sql(
    f"SELECT DISTINCT id, name, id_artist, acousticness, danceability, duration_ms, energy, instrumentalness, key, liveness, loudness, popularity, EXTRACT(year from `release_date`) as release_year, speechiness, tempo, valence FROM tracks ").toPandas()
users = spark.sql(f"SELECT user_id FROM users").toPandas()
tracks = pd.concat(
    [_tracks[['id', 'id_artist']], _tracks.drop(['id', 'name', 'id_artist'], axis=1).apply(stats.zscore)],
    axis=1)
sessions = spark.sql(
    """
    SELECT s.user_id, s.track_id, s.weight, acousticness, danceability, duration_ms, energy, instrumentalness, key, liveness, loudness, popularity, EXTRACT(year from `release_date`) as release_year, speechiness, tempo, valence
    FROM (
        select user_id, track_id, sum(event_weight) as weight
        from (
            SELECT user_id, track_id, 1 as event_weight
            FROM sessions
            WHERE event_type like 'like'
            ) 
        group by user_id, track_id
    ) s
    inner join tracks t on s.track_id = t.id
    order by s.user_id, t.id
    """).toPandas()

_tracks_with_artists = spark.sql(
    f"SELECT DISTINCT t.id as track_id, t.name as track_name, a.id as artist_id, a.name as artist_name FROM tracks t join artists a on t.id_artist == a.id").toPandas()

base_model = BaseModel(
    users_df=users,
    tracks_df=tracks,
    sessions_df=sessions
)


def choose_model(user_id: int) -> model_t:
    return 'base' if hash(user_id) % 2 == 0 else 'advanced'


def generate_recommendation_base(user_ids: list[int], playlist_length: int) -> tuple[pd.DataFrame, timedelta]:
    return _generate_recommendations('base', user_ids, playlist_length)


def generate_recommendation_advanced(user_ids: list[int], playlist_length: int) -> tuple[pd.DataFrame, timedelta]:
    return _generate_recommendations('base', user_ids, playlist_length)


def _generate_recommendations(model: model_t, user_ids: list[int], playlist_length: int) -> tuple[
    pd.DataFrame, timedelta]:
    start_time = time.time()
    if model == 'base':
        ids = base_model.predict_multiple(user_ids, playlist_length)
    elif model == 'advanced':
        ids = base_model.predict_multiple(user_ids, playlist_length)
    else:
        ids = []

    elapsed = time.time() - start_time

    return _tracks_with_artists.loc[_tracks_with_artists.track_id.isin(ids)].set_index('track_id')[
        ['track_name', 'artist_name']], elapsed


class PredictionHistoryRecord(TypedDict):
    user_id: int
    other_users: list[int]
    playlist: list[str]
    timestamp: datetime
    elapsed_time: timedelta
    model: model_t
    is_ab: bool


def dump_results(res: PredictionHistoryRecord):
    with open("predictions.jsonl", "a") as fp:
        json.dump(res, fp, default=str)
        fp.write('\n')
