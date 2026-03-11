import time
import requests

URL = "http://127.0.0.1:8000/generate"
PAYLOAD = {
    "prompt": "ROMEO:",
    "max_new_tokens": 30
}


def main():
    latencies = []
    num_requests = 10

    for _ in range(num_requests):
        start = time.time()
        response = requests.post(URL, json=PAYLOAD, timeout=30)
        response.raise_for_status()
        end = time.time()
        latencies.append(end - start)

    avg_latency = sum(latencies) / len(latencies)
    print(f"Average latency over {num_requests} requests: {avg_latency:.4f} seconds")


if __name__ == "__main__":
    main()