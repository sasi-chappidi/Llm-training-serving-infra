import requests

payload = {
    "prompt": "ROMEO:",
    "max_new_tokens": 50
}

response = requests.post("http://127.0.0.1:8000/generate", json=payload, timeout=30)
print(response.json())