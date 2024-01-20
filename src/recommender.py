import time
from typing import Literal
from datetime import timedelta

import pandas as pd

from .model.model_protocol import ModelProtocol


class Recommender:
    model_t = Literal['base', 'advanced']

    def __init__(self, base_model: ModelProtocol, advanced_model: ModelProtocol, tracks_with_artists_df: pd.DataFrame):
        self._base_model = base_model
        self._advanced_model = advanced_model
        self._tracks_with_artists_df = tracks_with_artists_df

    def generate_recommendation_base(self, user_ids: list[int], playlist_length: int) -> tuple[pd.DataFrame, timedelta]:
        return self._generate_recommendations('base', user_ids, playlist_length)

    def generate_recommendation_advanced(self, user_ids: list[int], playlist_length: int) -> tuple[
        pd.DataFrame, timedelta]:
        return self._generate_recommendations('advanced', user_ids, playlist_length)

    def _generate_recommendations(self, model: model_t, user_ids: list[int], playlist_length: int) -> tuple[
        pd.DataFrame, timedelta]:
        start_time = time.time()
        if model == 'base':
            ids = self._base_model.predict_multiple(user_ids, playlist_length)
        elif model == 'advanced':
            ids = self._advanced_model.predict_multiple(user_ids, playlist_length)
        else:
            ids = []

        elapsed = time.time() - start_time

        return self._tracks_with_artists_df.loc[self._tracks_with_artists_df.track_id.isin(ids)].set_index('track_id')[
            ['track_name', 'artist_name']], elapsed
