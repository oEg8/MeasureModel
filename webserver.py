import os
import platform
import time
import psutil
from datetime import datetime
from uuid import uuid4
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from posture_predictor_nn import NNPostureClassifier
# from sklearn.preprocessing import StandardScaler
import pickle

def log(text):
    now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {text}")

class ResponseMeasurement():
    def __init__(self, 
                  measurementID:str,
                  values:str,
                  target:str,
                  prediction:str,
                    ):
        self.measurementID = measurementID
        self.time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.values = values
        self.target = target
        self.prediction = prediction

    def as_dict(self):
        return {
            "measurementID" : self.measurementID,
            "time" : self.time,
            "values" : self.values,
            "target" : self.target,
            "prediction" : self.prediction
        }

def check_runtime_environment(data_file_name:str) -> bool:
    global file_path 
    file_path = os.path.join(os.getcwd(), "webserver_data", data_file_name)

    if not os.path.exists(file_path):
        os.makedirs(os.sep.join(file_path.split(os.sep)[:-1]), exist_ok=True)
        data = pd.DataFrame(columns=[
            "measurementID",
            "time",
            "values",
            "target",
            "prediction"
            ])
        data.to_csv(file_path, index=False)
    return True

def scale_values(values:list):
    log(f"Webserver: Scaling values")
    columns = [f"feature_{i}" for i in range(1, 211)]
    df = pd.DataFrame([values], columns=columns)
    return standardScaler.transform(df)
    
def inference(values:list):
    log(f"Webserver: Inferencing model")
    if max(values) <= 100:
        error_string = "empty_mat"
        log(f"Webserver: {error_string}")
        return error_string

    scaled_values = scale_values(values)
    scaled_values = np.array(scaled_values).reshape(1, -1) 
    print("scaled_values: ", scaled_values)
    print("len(scaled_values)): ", len(scaled_values))
    model_prediction = model.predict(scaled_values) 
    log(f"Webserver: Prediction: '{model_prediction}'")
    return model_prediction

def save_data(measurementID:str, values:list, target:str, prediction:str):
    global data
    log(f"Webserver: Saving ResponseMeasurement")
    response_measurement = ResponseMeasurement(
        measurementID = measurementID,
        values = str(values),
        target = target,
        prediction = prediction
    )
    data = pd.concat([data, pd.DataFrame([response_measurement.as_dict()])], ignore_index=True)
    data.to_csv(file_path, index=False)

start_time = time.time()
data_file_name = "measuredata.csv"
if check_runtime_environment(data_file_name):
    data = pd.read_csv(file_path)
    log(f"Data at {file_path}")
else:
    raise RuntimeError("No valid runtime environment")
model = NNPostureClassifier()
model_name = "nn_posture_model"
model.load_model(model_name=model_name)
input_dim = model.input_dim

pickle_path = "scalers/standard_scaler.pkl"
with open(pickle_path, "rb") as f:
    standardScaler = pickle.load(f)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/", methods=["GET"])
def status_endpoint():
    log(f"Webserver: Setting telemetry")
    telemetry = {
        "platform": platform.platform(),
        "uptime" : str(int(time.time() - start_time)),
        "cpu_load": str(int(psutil.cpu_percent(interval=1) * 10)),
        "memory": str(int(psutil.virtual_memory().percent)) ,
        "data_len" : str(len(data)),
        "data_size": str(int(os.path.getsize(file_path) // 1024)) + " kb"
    }
    log(f"Webserver: Serving response")
    return jsonify(telemetry), 200

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    global data
    log(f"Webserver: Reading request")
    json_data = request.get_json()

    log(f"Webserver: Check (for) value(s)")
    if not json_data or "values" not in json_data:
        error_string = "No values provided"
        log(f"Webserver: {error_string}")
        return jsonify({"error": error_string}), 400

    if len(json_data["values"]) == 0:
        error_string = "Empty values provided"
        log(f"Webserver: {error_string}")
        return jsonify({"error": error_string}), 400

    if len(json_data["values"]) != input_dim:
        error_string = f"Values is of length: {len(json_data['values'])} instead of models input dim: {input_dim}"
        log(f"Webserver: {error_string}")
        return jsonify({"error": error_string}), 400

    values = json_data.get("values", [])
    target = json_data.get("target", '')

    model_prediction = inference(values)

    save_data(
        measurementID = str(uuid4()), 
        values = values, 
        target = target,
        prediction = model_prediction
        )

    log(f"Webserver: Serving response")
    return jsonify({"prediction": model_prediction}),201
    
@socketio.on('connect')
def connect_event():
    log(f"Webserver: Client disconnected")

@socketio.on('disconnect')
def disconnect_event():
    log(f"Webserver: Client disconnected")

@socketio.on("inference")
def inference_event(data):
    values = data.get("values", [])
    
    model_prediction = inference(values)
    log(f"Webserver: Serving response")
    emit("message", {"prediction" : model_prediction})

if __name__ == "__main__":
    PORT = 8080
    socketio.run(app, host="0.0.0.0", port=PORT)