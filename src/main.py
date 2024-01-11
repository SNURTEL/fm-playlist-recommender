from typing import List, Sequence

from fastapi import FastAPI, responses, Query
import pandas as pd
import numpy as np
from scipy import stats

from src.model.base_model import BaseModel
from src.spark.config import views
from src.spark.create_session import create_session

app = FastAPI()

VIEWS = views("v3")
spark = create_session()

for view, file in VIEWS.items():
    df = spark.read.json(file)
    df.createOrReplaceTempView(view)

_tracks = spark.sql(
    f"SELECT DISTINCT id, id_artist, acousticness, danceability, duration_ms, energy, instrumentalness, key, liveness, loudness, popularity, EXTRACT(year from `release_date`) as release_year, speechiness, tempo, valence FROM tracks ").toPandas()
users = spark.sql(f"SELECT user_id FROM users").toPandas()
tracks = pd.concat([_tracks[['id', 'id_artist']], _tracks.drop(['id', 'id_artist'], axis=1).apply(stats.zscore)],
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

base_model = BaseModel(
    users_df=users,
    tracks_df=tracks,
    sessions_df=sessions
)


@app.get("/", response_class=responses.PlainTextResponse)
def root():
    return """IUM Task 3 Variant 2
By Tomasz Owienko and Anna Schäfer

Example:
http://127.0.0.1:8000/base_model/1?playlist_length=10&other_users=2,3,4
"""

@app.get("/base_model/{user_id}")
def read_item(user_id: int, playlist_length: int, other_users: str = None):
    if other_users:
        users_str = other_users.split(',')
        users = [int(id) for id in users_str]
    else:
        users = []
    users.insert(0, user_id)
    result = base_model.predict_multiple(users, playlist_length)
    return result
