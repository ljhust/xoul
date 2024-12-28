import requests

response = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions/",
    json={
        "model": "sophosympatheia/Midnight-Miqu-70B-v1.0-lora",
        "messages": [
            {"role": "user", "content": "once upon a time"},
        ],
        "temperature": 0.7,
        "max_tokens": 400,
        "stream": False
    }
)

print(response.json())