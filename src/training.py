from typing import TypedDict, Optional

import numpy as np
from lightfm import LightFM
from scipy.sparse import coo_matrix, csr_array

NUM_THREADS = 12
VERBOSE = True


class TrainingKwargs(TypedDict):
    learning_rate: float
    item_alpha: float
    user_alpha: float
    no_components: int
    epochs: int


class EvaluationResults(TypedDict):
    auc_train: float
    auc_test: float
    precision_train: float
    precision_test: float
    recall_train: float
    recall_test: float
    clustering: float


def train_lightfm(
        interactions: coo_matrix,
        item_features: csr_array,
        tk: TrainingKwargs,
        random_state: Optional[np.random.RandomState] = None
) -> LightFM:
    model = LightFM(
        loss='warp',
        learning_rate=tk["learning_rate"],
        item_alpha=tk["item_alpha"],
        user_alpha=tk["user_alpha"],
        no_components=tk["no_components"],
        random_state=random_state
    )
    model.fit(
        interactions=interactions,
        item_features=item_features,
        epochs=tk["epochs"],
        num_threads=NUM_THREADS,
        verbose=VERBOSE)

    return model
