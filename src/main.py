from fastapi import FastAPI, responses, Query
from src.functions import get_time_str, create_users_list_for_playlist, base_model_prediction, advanced_model_prediction, choose_model_and_predict, parse_data, save_to_jsonl

app = FastAPI()


@app.get("/", response_class=responses.PlainTextResponse)
def root():
    return """IUM Task 3 Variant 2
By Tomasz Owienko and Anna Schäfer

Example:
http://0.0.0.0:8000/predict/1?playlist_length=10&other_users=2,3,4
For base model:
http://0.0.0.0:8000/predict/base_model/1?playlist_length=10&other_users=2,3,4
For advanced model:
http://0.0.0.0:8000/predict/advanced_model/1?playlist_length=10&other_users=2,3,4

You can also use swagger under:
http://0.0.0.0:8000/docs#/

Results are saved in predictions.jsonl
"""


@app.get("/predict/{user_id}")
def read_item(user_id: int, playlist_length: int, other_users: str = None):
    current_time = get_time_str()
    users = create_users_list_for_playlist(user_id, other_users)
    playlist, elapsed_time, chosen_model = choose_model_and_predict(users, playlist_length)
    result = parse_data(user_id, users, playlist, current_time, elapsed_time, chosen_model, True)
    save_to_jsonl(result)
    return result


@app.get("/predict/base_model/{user_id}")
def read_item(user_id: int, playlist_length: int, other_users: str = None):
    current_time = get_time_str()
    users = create_users_list_for_playlist(user_id, other_users)
    playlist, elapsed_time = base_model_prediction(users, playlist_length)
    result = parse_data(user_id, users, playlist, current_time, elapsed_time, 'base', False)
    save_to_jsonl(result)
    return result


@app.get("/predict/advanced_model/{user_id}")
def read_item(user_id: int, playlist_length: int, other_users: str = None):
    current_time = get_time_str()
    users = create_users_list_for_playlist(user_id, other_users)
    playlist, elapsed_time = advanced_model_prediction(users, playlist_length)
    result = parse_data(user_id, users, playlist, current_time, elapsed_time, 'advanced', False)
    save_to_jsonl(result)
    return result
