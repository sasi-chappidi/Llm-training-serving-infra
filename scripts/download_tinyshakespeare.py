import os
import requests

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT_PATH = "data/raw/input.txt"

os.makedirs("data/raw", exist_ok=True)

response = requests.get(URL, timeout=30)
response.raise_for_status()

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(response.text)

print(f"Saved dataset to {OUT_PATH}")