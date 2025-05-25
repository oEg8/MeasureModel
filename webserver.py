import os
import platform
import time
import psutil
from datetime import datetime
from uuid import uuid4
import pandas as pd
from flask import Flask, request, jsonify, Response
from posture_predictor_nn import NNPostureClassifier

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

start_time = time.time()
data_file_name = "measuredata.csv"
if check_runtime_environment(data_file_name):
    print(file_path)
    data = pd.read_csv(file_path)
else:
    raise RuntimeError("No valid runtime environment")
model = NNPostureClassifier()
model.load_model(model_name="posture_model_at_2025-05-19_14-50")
input_dim = model.input_dim
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    global data
    print(f"Webserver: Reading request")
    json_data = request.get_json()
    if not json_data or "values" not in json_data:
        error_string = "No values provided"
        print(f"Webserver: {error_string}")
        return jsonify({"error": error_string}), 400
    
    print(f"Webserver: Adding values to dataframe")
    response_measurement = ResponseMeasurement(
        measurementID = uuid4(),
        values = json_data["values"],
        target = json_data.get("target", ""),
        prediction = '"',
    )
    row_int = len(data)
    data = pd.concat([data, pd.DataFrame([response_measurement.as_dict()])], ignore_index=True)
    data.to_csv(file_path, index=False)

    if len(json_data["values"]) != input_dim:
        error_string = f"Values is of lenght: {len(json_data['values'])} instead of models input dim: {input_dim}"
        print(f"Webserver: {error_string}")
        return jsonify({"error": error_string}), 400

    print(f"Webserver: Inferencing model")
    model_prediction = model.predict(response_measurement.values) 
    print(f"Webserver: Prediction: '{model_prediction}'")

    print(f"Webserver: Saving prediction to dataframe")
    data.at[row_int, "prediction"] = model_prediction
    data.to_csv(file_path, index=False)


    print(f"Webserver: Serving response")
    return jsonify({"prediction": model_prediction}),201

@app.route("/download", methods=["GET"])
def download_endpoint():
    print(f"Webserver: Exporting dataframe to csv")
    csv = data.to_csv(index=False)

    print(f"Webserver: Serving response")
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={data_file_name}"}
    )

@app.route("/status", methods=["GET"])
def status_endpoint():
    print(f"Webserver: Setting telemetry")
    telemetry = {
        "platform": platform.platform(),
        "uptime" : str(int(time.time() - start_time)),
        "cpu_load": str(int(psutil.cpu_percent(interval=1) * 10)),
        "memory": str(int(psutil.virtual_memory().percent)) ,
        "data_len" : str(len(data)),
        "data_size": str(int(os.path.getsize(file_path) // 1024)) + " kb"
    }
    print(f"Webserver: Serving response")
    return jsonify(telemetry), 200

if __name__ == "__main__":
    PORT = 8080
    app.run(host="0.0.0.0", port=PORT)