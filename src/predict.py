from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'models', 'cat_breed_model.pkl')
encoders_path = os.path.join(BASE_DIR, 'models', 'cat_breed_encoders.pkl')

model = joblib.load(model_path)
encoders = joblib.load(encoders_path)

cat_cols = list(encoders.keys())

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Standardize boolean-like strings
    for col in ['Neutered_or_spayed', 'Allowed_outdoor']:
        if col in data:
            val = str(data[col]).strip().capitalize()
            if val in ['True', 'False']:
                data[col] = val
    df = pd.DataFrame([data])
    # Encode categorical columns using saved encoders
    for col in cat_cols:
        if col in df:
            df[col] = encoders[col].transform(df[col].astype(str))
    pred = model.predict(df)
    return jsonify({'breed': pred[0]})

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    data = request.json  # Should be a list of dicts
    df = pd.DataFrame(data)
    for col in cat_cols:
        if col in df:
            df[col] = encoders[col].transform(df[col].astype(str))
    preds = model.predict(df)
    return jsonify({'breeds': preds.tolist()})

@app.route('/health', methods=['GET'])
def health():
    status = 'ok'
    details = {}
    # Check if model and encoders are loaded
    try:
        _ = model.feature_names_in_
        details['model'] = 'loaded'
    except Exception:
        status = 'error'
        details['model'] = 'not loaded'
    try:
        _ = list(encoders.keys())
        details['encoders'] = 'loaded'
    except Exception:
        status = 'error'
        details['encoders'] = 'not loaded'
    return jsonify({'status': status, 'details': details})

@app.route('/features', methods=['GET'])
def features():
    # Get feature names from the trained model
    feature_list = list(model.feature_names_in_)
    return jsonify({'features': feature_list})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
