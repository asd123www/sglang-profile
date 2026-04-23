import sys
import requests

# Port 30000 for unified mode, 8000 for PD disaggregation (router)
port = int(sys.argv[1]) if len(sys.argv) > 1 else 30000

response = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={
    "model": "lmsys/gpt-oss-120b-bf16",
    "messages": [{"role": "user", "content": "Hello, can you tell me who you are? What is the name of your model?"}],
    "max_tokens": 256,
})
print(response.json()["choices"][0]["message"]["content"])
