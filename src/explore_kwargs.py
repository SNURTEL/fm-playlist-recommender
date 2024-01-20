import warnings
import csv
import time
from datetime import datetime
from typing import TypedDict
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy import stats
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k, reciprocal_rank


from src.spark.config import views
from src.spark.create_session import create_session
from src.training import TrainingKwargs, train_lightfm
from src.model.base_model import BaseModel
from src.model.fm_model import FMModel

"""
Test random hyperparams.
Run from `src` directory.
"""

DATA_VERSION = "v4"
RESULTS_FILE = str(datetime.now()) + "training_log.tsv"
RANDOM_SEED = 12345678

CLUSTERING_NO_REPS = 400
PLAYLIST_LENGTH = 20


class EvaluationResults(TypedDict):
    auc_train: float
    auc_test: float
    precision_train: float
    precision_test: float
    recall_train: float
    recall_test: float
    reciprocal_rank_train: float
    reciprocal_rank_test: float
    clustering_fm: float
    clustering_base: float
    percent_better: float
    time_elapsed: timedelta


def mean_dist_from_cluster_center(items_ids, model: LightFM, item_id_mapping: dict):
    item_indices = [item_id_mapping[i] for i in items_ids]
    coords = np.take(model.get_item_representations()[1], item_indices, axis=0)
    center = np.sum(coords, axis=0) / coords.shape[0]
    return np.average(np.apply_along_axis(lambda x: np.linalg.norm(center - x, ord=2), 1, coords))



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

    (user_id_mapping, _, item_id_mapping, _) = dataset.mapping()

    (train, test) = random_train_test_split(interactions)

    feature_names = tracks.drop(['id'], axis=1).columns

    item_features = dataset.build_item_features(
        ((i, feature_names) for i in tracks['id']),
        normalize=False)
    print(f"Item feature matrix: {repr(item_features)}")

    # ======== TRAIN AND EVALUATE ========

    with open(RESULTS_FILE, mode='w') as fp:
        w = csv.writer(fp, delimiter='\t', lineterminator='\n')
        w.writerow(list(TrainingKwargs.__annotations__.keys()) + list(EvaluationResults.__annotations__.keys()))

    i = 1
    while True:
        training_kwargs: TrainingKwargs = {
            "learning_rate": 10 ** np.random.uniform(low=-5, high=-0.7),
            "item_alpha": 10 ** np.random.uniform(low=-6, high=-1),
            "user_alpha": 10 ** np.random.uniform(low=-6, high=-1),
            "no_components": np.random.randint(low=5, high=80),
            "epochs": 10,
        }

        print(f"[{i}] train with {training_kwargs}")
        time_started = time.time()
        lightfm_model = train_lightfm(train, item_features, training_kwargs, random_state=np.random.RandomState(RANDOM_SEED))
        time_elapsed = time.time() - time_started

        fm_model = FMModel(
            users_df=users,
            tracks_df=tracks,
            sessions_df=d,
            lightfm_model=lightfm_model,
            interactions=interactions,
            item_features=item_features,
            user_id_mapping=user_id_mapping
        )

        base_model = BaseModel(
            users_df=users,
            tracks_df=tracks,
            sessions_df=d
        )

        (_, _, item_id_mapping, _) = dataset.mapping()

        print(f"[{i}] 0/10 done...", end="")
        train_auc = auc_score(lightfm_model, train,
                                item_features=item_features,
                                num_threads=12).mean()
        print(f"\r[{i}] 1/10 done...", end="")
        test_auc = auc_score(lightfm_model, test,
                               train_interactions=train,
                               item_features=item_features,
                               num_threads=12).mean()
        print(f"\r[{i}] 2/10 done...", end="")

        train_precision = precision_at_k(lightfm_model, train, k=10,
                                           item_features=item_features,
                                           num_threads=12).mean()
        print(f"\r[{i}] 3/10 done...", end="")
        test_precision = precision_at_k(lightfm_model, test, k=10,
                                          train_interactions=train,
                                          item_features=item_features,
                                          num_threads=12).mean()
        print(f"\r[{i}] 4/10 done...", end="")

        train_recall = recall_at_k(lightfm_model, train, k=10,
                                     item_features=item_features,
                                     num_threads=12).mean()
        print(f"\r[{i}] 5/10 done...", end="")
        test_recall = recall_at_k(lightfm_model, test, k=10,
                                    train_interactions=train,
                                    item_features=item_features,
                                    num_threads=12).mean()
        print(f"\r[{i}] 6/10 done...", end="")

        train_reciprocal_rank = reciprocal_rank(lightfm_model, train,
                                                  item_features=item_features,
                                                  num_threads=12).mean()
        print(f"\r[{i}] 7/10 done...", end="")
        test_reciprocal_rank = reciprocal_rank(lightfm_model, test,
                                                 train_interactions=train,
                                                 item_features=item_features,
                                                 num_threads=12).mean()
        print(f"\r[{i}] 8/10 done...", end="")

        mean_dists_fm = []
        for _ in range(CLUSTERING_NO_REPS):
            no_users = np.random.randint(low=2, high=10)
            user_ids = users['user_id'].sample(no_users).tolist()
            r = fm_model.predict_multiple(user_ids, PLAYLIST_LENGTH)
            mean_dists_fm.append(mean_dist_from_cluster_center(r, lightfm_model, item_id_mapping))

        mean_dist_fm = sum(mean_dists_fm) / CLUSTERING_NO_REPS

        print(f"\r[{i}] 9/10 done...", end="")

        mean_dists_base = []
        for _ in range(CLUSTERING_NO_REPS):
            no_users = np.random.randint(low=2, high=10)
            user_ids = users['user_id'].sample(no_users).tolist()
            r = base_model.predict_multiple(user_ids, PLAYLIST_LENGTH)
            mean_dists_base.append(mean_dist_from_cluster_center(r, lightfm_model, item_id_mapping))

        mean_dist_base = sum(mean_dists_base) / CLUSTERING_NO_REPS

        print(f"\r[{i}] 10/10 done...")

        res: EvaluationResults = {
            "auc_train": train_auc,
            "auc_test": test_auc,
            "precision_train": train_precision,
            "precision_test": test_precision,
            "recall_train": train_recall,
            "recall_test": test_recall,
            "reciprocal_rank_train": train_reciprocal_rank,
            "reciprocal_rank_test": test_reciprocal_rank,
            "clustering_fm": mean_dist_fm,
            "clustering_base": mean_dist_base,
            "percent_better": (mean_dist_base - mean_dist_fm) * 100 / mean_dist_base,
            "time_elapsed": time_elapsed
        }

        print(f"[{res}]")

        with open(RESULTS_FILE, mode='a') as fp:
            w = csv.writer(fp, delimiter='\t', lineterminator='\n')
            w.writerow(list(training_kwargs.values()) + list(res.values()))

        i += 1


if __name__ == '__main__':
    main()
