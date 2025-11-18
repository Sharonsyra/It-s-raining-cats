from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('../models/cat_breed_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    pred = model.predict(df)
    return jsonify({'breed': pred[0]})

if __name__ == '__main__':
    app.run(port=5000)
