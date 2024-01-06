from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd
import numpy as np
from lightfm import LightFM
from scipy.sparse import csr_array


class FMModel:
    """
    A wrapper on LightFM model class allowing to make multi-user predictions. The interaction matrix, feature matrices
    and the LightFM model itself have to be prepared separately (check example notebook).
    """
    def __init__(self,
                 users_df: pd.DataFrame,  # needs to have an `id` column
                 tracks_df: pd.DataFrame,  # needs to have an `id` column
                 sessions_df: pd.DataFrame,
                 lightfm_model: LightFM,
                 interactions: csr_array,
                 item_features: Optional[csr_array] = None,
                 user_features: Optional[csr_array] = None):
        self.model = lightfm_model
        self.interactions = interactions
        self.item_features = item_features
        self.user_features = user_features  # unused for now
        self.users = users_df  # unused for now
        self.tracks = tracks_df
        self.sessions = sessions_df

    @staticmethod
    def from_serialized(self, j: str) -> FMModel:
        raise NotImplementedError("Deserialization not implemented")

    def predict_single(self, user: int, number: int, num_threads=4) -> pd.Series:
        return self._predict_single(user=user, number=number, num_threads=num_threads).reset_index(drop=True)

    def _predict_single(self, user: int, number: int, num_threads=4) -> pd.Series:
        predicted_scores = self.model.predict(user, np.arange(self.interactions.shape[1]),
                                              item_features=self.item_features,
                                              num_threads=num_threads)

        return pd.concat([self.tracks['id'], pd.Series(predicted_scores)], axis=1) \
            .nlargest(number, 0) \
            .sort_values(by=0, ascending=False)['id']

    def predict_multiple(self, users: Sequence[int], number: int) -> pd.Series:
        if len(users) <= 1:
            raise ValueError("To make single-user predictions, use `predict_single`")

        predictions = [self.predict_single(u, self.tracks.size) for u in users]
        common = frozenset(predictions[0]).intersection(*(frozenset(p) for p in predictions[1:]))

        indexed_by_rank = (s.reset_index(drop=True).loc[s.reset_index().id.isin(common)].sort_values() for s in
                           predictions)

        return pd.Series(pd.concat([pd.Series(s.index.values, index=s) for s in indexed_by_rank], axis=1).mean(
            'columns').nsmallest(number).index.values)
