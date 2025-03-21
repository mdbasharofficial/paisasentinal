from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

# Load the trained XGBoost model
model = joblib.load('xgboost_fraud_model.joblib')

# Rule Engine
class RuleEngine:
    def __init__(self):
        self.rules = [
            lambda t: t['transaction_amount'] > 1000,
            lambda t: not (9 <= datetime.fromisoformat(t['transaction_date']).hour < 17)
        ]

    def evaluate(self, transaction):
        return any(rule(transaction) for rule in self.rules)

rule_engine = RuleEngine()

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/evaluate', methods=['POST'])
def evaluate_transaction():
    transaction = request.json
    
    # Rule-based detection
    is_fraud_rule = rule_engine.evaluate(transaction)
    
    # AI model detection
    features = pd.DataFrame([transaction])
    features = features.drop(['transaction_id', 'transaction_date'], axis=1, errors='ignore')
    fraud_score = model.predict_proba(features)[0][1]
    is_fraud_model = fraud_score > 0.5
    
    is_fraud = is_fraud_rule or is_fraud_model
    
    response = {
        "transaction_id": transaction.get('transaction_id', ''),
        "is_fraud": is_fraud,
        "fraud_source": "rule" if is_fraud_rule else "model" if is_fraud_model else None,
        "fraud_reason": "Rule-based detection" if is_fraud_rule else "AI model prediction" if is_fraud_model else None,
        "fraud_score": float(fraud_score)
    }
    
    # Here you would typically store the result in a database
    # store_result_in_database(transaction, response)
    
    return jsonify(response), 200

@app.route('/add_rule', methods=['POST'])
def add_rule():
    rule_type = request.json['type']
    params = request.json['params']
    
    if rule_type == 'amount_threshold':
        rule_engine.rules.append(lambda t: t['transaction_amount'] > params['threshold'])
    elif rule_type == 'unusual_time':
        rule_engine.rules.append(lambda t: not (params['start_hour'] <= datetime.fromisoformat(t['transaction_date']).hour < params['end_hour']))
    
    return jsonify({"message": "Rule added successfully"}), 200

@app.errorhandler(404)
def page_not_found(e):
    return "404 - Page Not Found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
