from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model_rf_multioutput.joblib")
scaler_X = joblib.load("scaler_X.joblib")
scaler_y = joblib.load("scaler_y.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    lags = data.get("lags")
    if lags is None or len(lags) != 14:
        return {"error": "provide 'lags' list of length 14 (lag_1...lag_14)"}, 400
    lags = np.array(lags).reshape(1, -1)
    Xs = scaler_X.transform(lags)
    y_s = model.predict(Xs)
    y = scaler_y.inverse_transform(y_s)
    return jsonify({"forecast_7days": y.tolist()[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)