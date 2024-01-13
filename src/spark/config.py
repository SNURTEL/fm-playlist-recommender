from typing import Optional


def views(version: str, filter_items: Optional[list[str]] = None) -> dict[str, str]:
    v = {
        'artists': f"../data/{version}/artists.jsonl",
        'sessions': f"../data/{version}/sessions.jsonl",
        'track_storage': f"../data/{version}/track_storage.jsonl",
        'tracks': f"../data/{version}/tracks.jsonl",
        'users': f"../data/{version}/users.jsonl",
    }
    return v if filter_items is None else {k: v for k, v in v.items() if k in filter_items}


NUMBER_COLUMNS = {
    'artists': [],
    'sessions': ['session_id',  'user_id'],
    'track_storage': ['daily_cost'],
    'tracks': [
        'acousticness',
        'danceability',
        'duration_ms',
        'energy',
        'explicit',
        'instrumentalness',
        'key',
        'liveness',
        'loudness',
        'popularity',
        'speechiness',
        'tempo',
        'valence'
    ],
    'users': ['user_id'],
}


LIST_COLUMNS = {
    'artists': ['genres'],
    'sessions': [],
    'track_storage': [],
    'tracks': [],
    'users': ['favourite_genres'],
}

DATE_FORMAT = "yyyy-MM-dd'T'HH:mm:ss[.SSSSSS]"
