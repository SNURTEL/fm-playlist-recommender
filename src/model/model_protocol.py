from typing import Protocol, Sequence

import pandas as pd


class Recommender(Protocol):
    def predict_single(self, user: int, number: int) -> pd.Series:
        """
        Recommends `number` of tracks to user of ID `user`.

        Args:
            user (int): user ID
            number (int): number of recommendations to make

        Returns:
             pd.Series: Predicted tracks. Series values are track ID's.
        """
        ...

    def predict_multiple(self, users: Sequence[int], number: int) -> pd.Series:
        """
        Recommend `number` of tracks to users of ID's `users`.

        Args:
            users (list[int]): user ID's (at least two)
            number (int): number of recommendations to make

        Returns:
             pd.Series: Predicted tracks. Series values are track ID's, ordered by mean rank in single-user predictions.

        Raises:
            ValueError: No users supplied.
        """
        ...

