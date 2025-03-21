from flask import Flask, request, jsonify
from datetime import datetime
import json
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained XGBoost model
model = joblib.load('xgboost_fraud_model_tuned.joblib')

class RuleEngine:
    def __init__(self):
        self.rules = []

    def add_rule(self, rule):
        self.rules.append(rule)

    def evaluate(self, transaction):
        results = []
        for rule in self.rules:
            if rule.evaluate(transaction):
                results.append(rule.message)
        return results

class Rule:
    def __init__(self, condition, message):
        self.condition = condition
        self.message = message

    def evaluate(self, transaction):
        return self.condition(transaction)

def amount_threshold_rule(threshold):
    return Rule(
        lambda transaction: transaction['transaction_amount'] > threshold,
        f"Transaction amount exceeds {threshold}"
    )

def unusual_time_rule(start_hour, end_hour):
    return Rule(
        lambda transaction: not (start_hour <= datetime.fromisoformat(transaction['transaction_date']).hour < end_hour),
        f"Transaction occurred outside normal hours ({start_hour}:00 - {end_hour}:00)"
    )

# Initialize the rule engine
engine = RuleEngine()

# Add some initial rules
engine.add_rule(amount_threshold_rule(1000))
engine.add_rule(unusual_time_rule(9, 17))

def ai_model_prediction(transaction):
    # Prepare the input for the model
    features = [
        transaction['transaction_amount'],
        transaction['transaction_hour'],
        transaction['transaction_day'],
        transaction['transaction_month'],
        transaction['transaction_dayofweek'],
        # Add other relevant features here
    ]
    features = np.array(features).reshape(1, -1)
    
    # Make prediction
    fraud_score = model.predict_proba(features)[0][1]
    is_fraud = fraud_score > 0.5  # Adjust threshold as needed
    
    return is_fraud, fraud_score

def store_result_in_database(transaction, result):
    # TODO: Implement database storage
    pass

@app.route('/add_rule', methods=['POST'])
def add_rule():
    rule_type = request.json['type']
    params = request.json['params']
    
    if rule_type == 'amount_threshold':
        engine.add_rule(amount_threshold_rule(params['threshold']))
    elif rule_type == 'unusual_time':
        engine.add_rule(unusual_time_rule(params['start_hour'], params['end_hour']))
    
    return jsonify({"message": "Rule added successfully"}), 200

@app.route('/evaluate', methods=['POST'])
def evaluate_transaction():
    transaction = request.json
    rule_results = engine.evaluate(transaction)
    ai_fraud, ai_score = ai_model_prediction(transaction)
    
    is_fraud = len(rule_results) > 0 or ai_fraud
    fraud_source = "rule" if len(rule_results) > 0 else "model" if ai_fraud else None
    fraud_reason = "; ".join(rule_results) if len(rule_results) > 0 else "AI model prediction" if ai_fraud else None
    
    response = {
        "transaction_id": transaction.get('transaction_id', ''),
        "is_fraud": is_fraud,
        "fraud_source": fraud_source,
        "fraud_reason": fraud_reason,
        "fraud_score": float(ai_score)
    }
    
    store_result_in_database(transaction, response)
    
    return jsonify(response), 200

@app.route('/')
def home():
    return "Fraud Detection API is running"

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"error": "404 - Not Found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
