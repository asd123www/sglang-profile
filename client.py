import requests

response = requests.post("http://127.0.0.1:30000/v1/chat/completions", json={
    "model": "lmsys/gpt-oss-120b-bf16",
    "messages": [{"role": "user", "content": "Hello, can you tell me who you are? What is the name of your model?"}],
    "max_tokens": 256,
})
print(response.json()["choices"][0]["message"]["content"])
