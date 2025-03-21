import requests

url = "http://127.0.0.1:5001/evaluate"
sample_transaction = {
    "transaction_id": "TXN12345",
    "transaction_amount": 1500,
    "transaction_date": "2025-03-21T13:47:00",
    "transaction_channel": "web",
    "transaction_payment_mode": "card",
    "payment_gateway_bank": "BankA",
    "payer_email": "payer@example.com",
    "payer_mobile": "1234567890",
    "payer_card_brand": "Visa",
    "payer_device": "Device123",
    "payer_browser": "Chrome",
    "payee_id": "PAYEE123"
}

response = requests.post(url, json=sample_transaction)
print(response.json())
