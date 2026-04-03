from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

@app.route("/")
def home():
    return "API OK"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["data"]

        # encode 3 cột string KDD
        data[1] = encoders["protocol_type"].transform([data[1]])[0]
        data[2] = encoders["service"].transform([data[2]])[0]
        data[3] = encoders["flag"].transform([data[3]])[0]

        # bỏ label + difficulty nếu user gửi nhầm
        if len(data) > 41:
            data = data[:41]

        features = np.array(data).astype(float).reshape(1, -1)

        result = model.predict(features)[0]

        return jsonify({"result": str(result)})

    except Exception as e:
        return jsonify({"error": str(e)})
