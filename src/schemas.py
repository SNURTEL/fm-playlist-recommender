from pydantic import BaseModel, field_validator


class PredictRequestSchema(BaseModel):
    user_id: int
    other_users: list[int]
    playlist_length: int

    @field_validator("playlist_length")
    @classmethod
    def validate_playlist_length(cls, v: int):
        if v < 0:
            return "Playlist length cannot be negative"
        return v

    @property
    def users(self):
        return list(set([self.user_id] + self.other_users))


class PredictionListEntrySchema(BaseModel):
    id: str
    name: str
    artist: str
