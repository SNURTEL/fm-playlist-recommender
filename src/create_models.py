import warnings
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split

from src.spark.config import views
from src.spark.create_session import create_session
from src.training import TrainingKwargs, train_lightfm
from src.model.base_model import BaseModel
from src.model.fm_model import FMModel

"""
Train and serialize both base and advanced models.
Run from `src` directory.
"""

DATA_VERSION = "v4"
SERIALIZED_DIR = "serialized"
TRAIN_KWARGS: TrainingKwargs = {
    "learning_rate": 0.05,
    "item_alpha": 1e-6,
    "user_alpha": 1e-6,
    "no_components": 30,
    "epochs": 10,
}
RANDOM_SEED = 12345678


def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)

    VIEWS = views(DATA_VERSION)
    spark = create_session()

    for view, file in VIEWS.items():
        df = spark.read.json(file)
        df.createOrReplaceTempView(view)

    _tracks = spark.sql(
        f"SELECT DISTINCT id, id_artist, acousticness, danceability, duration_ms, energy, instrumentalness, key, liveness, loudness, popularity, EXTRACT(year from `release_date`) as release_year, speechiness, tempo, valence FROM tracks ").toPandas()
    tracks = pd.concat([_tracks[['id', 'id_artist']], _tracks.drop(['id', 'id_artist'], axis=1).apply(stats.zscore)],
                       axis=1)
    users = spark.sql(f"SELECT user_id FROM users").toPandas()

    d = spark.sql(
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

    dataset = Dataset()
    dataset.fit(
        users=users['user_id'],
        items=tracks['id']
    )
    dataset.fit_partial(
        items=tracks['id'],
        item_features=tracks.drop('id', axis=1)
    )
    num_users, num_items = dataset.interactions_shape()
    print('Num users: {}, num_items {}.'.format(num_users, num_items))

    (interactions, weights) = dataset.build_interactions(d[['user_id', 'track_id']].apply(tuple, axis=1))
    print(f"Interaction matrix: {repr(interactions)}")

    (train, test) = random_train_test_split(interactions)

    feature_names = tracks.drop(['id'], axis=1).columns

    item_features = dataset.build_item_features(
        ((i, feature_names) for i in tracks['id']),
        normalize=False)
    print(f"Item feature matrix: {repr(item_features)}")

    lightfm_model = train_lightfm(train, item_features, TRAIN_KWARGS, random_state=np.random.RandomState(RANDOM_SEED))

    fm_model = FMModel(
        users_df=users,
        tracks_df=tracks,
        sessions_df=d,
        lightfm_model=lightfm_model,
        interactions=interactions,
        item_features=item_features,
    )

    base_model = BaseModel(
        users_df=users,
        tracks_df=tracks,
        sessions_df=d
    )

    with open(f"{SERIALIZED_DIR}/base_model.bin", mode='wb') as fp:
        fp.write(pickle.dumps(base_model))

    with open(f"{SERIALIZED_DIR}/advanced_model.bin", mode='wb') as fp:
        fp.write(pickle.dumps(fm_model))

    print(f"Models saved to `{SERIALIZED_DIR}`")


if __name__ == '__main__':
    main()
