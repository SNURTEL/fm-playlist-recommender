from typing import List, Sequence
from model.base_model import BaseModel
import pandas as pd
import numpy as np
from scipy import stats
import time
import json
from spark.config import views
from spark.create_session import create_session


VIEWS = views("v4")
spark = create_session()

for view, file in VIEWS.items():
    df = spark.read.json(file)
    df.createOrReplaceTempView(view)

_tracks = spark.sql(
    f"SELECT DISTINCT id, id_artist, acousticness, danceability, duration_ms, energy, instrumentalness, key, liveness, loudness, popularity, EXTRACT(year from `release_date`) as release_year, speechiness, tempo, valence FROM tracks ").toPandas()
users = spark.sql(f"SELECT user_id FROM users").toPandas()
tracks = pd.concat([_tracks[['id', 'id_artist']], _tracks.drop(['id', 'id_artist'], axis=1).apply(stats.zscore)], axis=1)
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

base_model = BaseModel(
    users_df=users,
    tracks_df=tracks,
    sessions_df=sessions
)


def get_time_str():
    now = time.localtime()
    current_time = time.strftime("%y-%m-%dT%H:%M:%S", now)
    return current_time


def create_users_list_for_playlist(user_id: int, other_users: str = None):
    if other_users:
        users_str = other_users.split(',')
        users = [int(id) for id in users_str]
    else:
        users = []
    users.insert(0, user_id)
    return users


def base_model_prediction(users: List, playlist_length: int):
    start_time = time.time()
    playlist = base_model.predict_multiple(users, playlist_length)
    end_time = time.time()
    return playlist, end_time - start_time


def advanced_model_prediction(users: List, playlist_length: int):
    start_time = time.time()
    # TODO CHANGE MODEL TO ADVANCED
    playlist = base_model.predict_multiple(users, playlist_length)
    end_time = time.time()
    return playlist, end_time - start_time


def choose_model_and_predict(users: List, playlist_length: int):
    if hash(users[0]) % 2 == 0:
        chosen_model = 'base'
        playlist, elapsed_time = base_model_prediction(users, playlist_length)
    else:
        chosen_model = 'advanced'
        playlist, elapsed_time = advanced_model_prediction(users, playlist_length)
    return playlist, elapsed_time, chosen_model


def parse_data(user_id: int, users_id: List, playlist_series, timestamp, elapsed_time, model_type: str, is_random: bool):
    # playlist_series is pd.Series
    if len(users_id) == 1:
        other_users = None
    else:
        other_users = [id for id in users_id]
        other_users.remove(user_id)
    dict_data = {
        "user_id": user_id,
        "other_users": other_users,
        "playlist": playlist_series.values.tolist(),
        "timestamp": timestamp,
        "elapsed_time": elapsed_time,
        "model_type": model_type,
        "is_random": is_random
    }
    return dict_data


def save_to_jsonl(data):
    json_object = json.dumps(data)
 
    with open("predictions.jsonl", "a") as file_handle:
        file_handle.write(json_object)
        file_handle.write('\n')