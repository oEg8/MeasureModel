from flask import Flask, request, jsonify
from posture_predictor_nn import NNPostureClassifier

model = NNPostureClassifier()
model.load_model(model_name="posture_model_at_2025-05-19_14-50")

app = Flask(__name__)
@app.route("/pm_predict", methods=["POST"])
def predict_endpoint():
    json_data = request.get_json()
    if not json_data or "input" not in json_data:
        print(f"Webserver: No input data provided")
        return jsonify({"error": "No input data provided"}), 400
    
    input_data = json_data["input"]
    print(f"Webserver: Request for: {input_data}")
    
    result = model.predict(input_data)
    print(f"Webserver: Prediction: {result}")

    print(f"Webserver: Serving response")
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True, host="localhost")