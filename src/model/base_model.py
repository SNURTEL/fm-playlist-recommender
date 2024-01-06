from typing import Sequence
import pandas as pd


class BaseModel:
    """
    Base model. Makes predictions by drawing random samples from users' session history.
    """
    def __init__(self, users_df: pd.DataFrame, tracks_df: pd.DataFrame, sessions_df: pd.DataFrame):
        self.users = users_df  # needs to have an `id` column
        self.tracks = tracks_df  # needs to have an `id` column
        self.sessions = sessions_df  # needs to have `track_id` and `user_id` columns

    def predict_single(self, user: int, number: int) -> pd.Series:
        return self.sessions.loc[self.sessions['user_id'] == user]['track_id'].sample(number).reset_index(drop=True)

    def predict_multiple(self, users: Sequence[int], number: int) -> pd.Series:
        return pd.concat(
            (self.sessions.loc[self.sessions['user_id'] == i] for i in users)
        )['track_id'].sample(number).reset_index(drop=True)
