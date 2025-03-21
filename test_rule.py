import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import requests
from datetime import datetime

# Load the data
df = pd.read_csv('preprocessed_data.csv')

# Function to evaluate a single transaction
def evaluate_transaction(transaction):
    url = "http://localhost:5001/evaluate"
    response = requests.post(url, json=transaction)
    if response.status_code == 200:
        return response.json()['is_fraud']
    else:
        raise Exception(f"API request failed with status code {response.status_code}")

# Evaluate all transactions
predictions = []
for _, transaction in df.iterrows():
    prediction = evaluate_transaction(transaction.to_dict())
    predictions.append(prediction)

# Calculate performance metrics
accuracy = accuracy_score(df['is_fraud'], predictions)
precision = precision_score(df['is_fraud'], predictions)
recall = recall_score(df['is_fraud'], predictions)
f1 = f1_score(df['is_fraud'], predictions)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
