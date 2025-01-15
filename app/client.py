import requests

response = requests.post(
    "http://localhost:8000/mixtral/invoke",
    json={'input': {'topic': 'cats'}}
)
print(response.json())