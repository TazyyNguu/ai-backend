from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# load model + encoder
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

# nếu có scaler
try:
    scaler = joblib.load("scaler.pkl")
except:
    scaler = None

@app.route("/")
def home():
    return "API OK"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["data"]

        # chuyển list → numpy
        features = np.array(data).reshape(1, -1)

        # nếu có scaler
        if scaler:
            features = scaler.transform(features)

        prediction = model.predict(features)[0]

        return jsonify({
            "result": str(prediction)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":
    app.run()
