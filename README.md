# Posture Prediction from Foot Pressure

This project aims to predict human posture based on pressure distribution from a footprint. Two approaches are explored:

* **Traditional Machine Learning** (RandomForestClassifier)
* **Neural Networks** (via PyTorch)

The model uses real-world pressure sensor readings.


## Project Structure   --- UPDATE AT END OF PROJECT!! ---

```
MeasureModel/
├── main.py                         # Main entry point for running predictions
├── posture_predictor_ml.py         # Traditional ML posture prediction
├── posture_predictor_nn.py         # Neural Network posture prediction
├── scaler_generator.py             # Script to generate and save scalers
├── webserver.py                    # (Optional) Web interface or API server
├── data/                           # All raw and processed datasets
│   ├── data_converter.py           # Script to convert/clean raw data
│   ├── raw/
│   │   ├── initial_data.csv        # Original dataset
│   │   └── new_data.csv            # Newly collected data
│   └── processed/
│       └── final_combined.csv      # Merged and cleaned dataset
├── models/                         # Trained model files
│   ├── ml_posture_model.pkl        # Traditional ML model
│   ├── nn_posture_model.safetensors# Neural network weights
│   └── nn_posture_model.json       # NN model architecture
├── scalers/                        # Preprocessing scalers
│   └── standard_scaler.pkl         # StandardScaler object
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
└── .gitignore                      # Git ignore file

```


## Getting Started

### 1. Clone the repository

```bash
git clone '' ###
cd measuremodel
```

### 2. Create and activate the environment

When using Anaconda:

```bash
conda create -n yourname python=3.10
conda activate yourname
```

When using venv:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

Install required libraries including `pandas`, `scikit-learn`, and `torch`:

```bash
pip install -r requirements.txt
```

## Usage

To run the model and make predictions run the main.py. Specify the data that you want to be used and the machine learning type. 

By default, the script initializes and trains the neural network model using actual observations, then prints diffrent evaluation metrics.


## Current Status

* ML classifier (RandomForestClassifier) implemented and functional.
* Neural network architecture defined using PyTorch.


## Example Input

Data input consists of 210 pressure features (foot pressure points), and a corresponding posture label (e.g., "correct_posture", "inbalance_left", "on_toes").


## Planned Improvements

* Expansion of datapoints.
* Expansion of posture labels (e.g. slouching, asymmetrical stance).


## Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request with ideas, bug fixes, or improvements.


## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.