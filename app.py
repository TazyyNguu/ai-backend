from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# load model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "API OK"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["data"]

        # chuyển về numpy
        features = np.array(data).reshape(1, -1)

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
