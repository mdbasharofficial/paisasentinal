import requests

url = "http://127.0.0.1:5000/add_rule"
payload = {
    "type": "amount_threshold",
    "params": {
        "threshold": 1000
    }
}

response = requests.post(url, json=payload)
print(response.json())
