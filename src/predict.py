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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(port=5000)
