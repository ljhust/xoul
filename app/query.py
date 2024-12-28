import json
import requests

def get_response(query: str, model_name: str):
    response = requests.post(
        "http://127.0.0.1:8000/v1/chat/completions/",
        json={
            "model": model_name,
            "messages": [
                {"role": "user", "content": query},
            ],
            "temperature": 0.7,
            "max_tokens": 400,
            "stream": False
        }
    )
    
    # json to string for saving
    json_response = json.dumps(response.json())
    return json_response  

