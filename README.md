# Posture Prediction from Foot Pressure

This project aims to predict human posture based on pressure distribution from a footprint. Two approaches are explored:

* **Traditional Machine Learning** (e.g. XGBoost)
* **Neural Networks** (via PyTorch)

The model currently uses **synthetically generated data**, but it is built with the goal of eventually processing **real-world pressure sensor readings**.


## 📁 Project Structure

```
MeasureModel/
├── main.py                       # Main entry point for running predictions
├── posture_predictor_ml.py       # Traditional ML implementation
├── posture_predictor_nn.py       # Neural Network implementation
├── data_generator.py             # Random data generator for test cases
├── models/                       # Saved models (once trained)
├── README.md                     # Project documentation
└──  requirements.txt              # Dependencies incl. version
```


## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/posture-predictor.git
cd posture-predictor
```

### 2. Create and activate the environment

We recommend using Anaconda:

```bash
conda create -n sem6 python=3.10
conda activate sem6
```

### 3. Install dependencies

Install required libraries including `xgboost`, `scikit-learn`, and `torch`:

```bash
pip install -r requirements.txt
```

If you encounter issues with `xgboost` on macOS (e.g. OpenMP errors), see the troubleshooting section below.


## ⚙️ Usage

To run the model and make predictions using randomly generated data:

```bash
python main.py
```

By default, the script initializes and trains both the ML and neural network models using synthetic data, then prints predictions.


## 🧪 Current Status

* ✅ ML classifier (XGBoost) implemented and functional.
* ✅ Neural network architecture defined using PyTorch.
* 🔄 Models train on randomly generated synthetic input.
* 🚧 Real-world data integration in progress.


## 🗃 Example Input

Synthetic input consists of 144 pressure features (e.g., from a foot pressure mat), and a corresponding posture label (e.g., "straight", "leaning left", "leaning forward").


## 📊 Planned Improvements

* Integration with real foot pressure sensors.
* Expansion of posture labels (e.g. slouching, asymmetrical stance).
* Live classification via camera + sensor input.
* Visualization of foot pressure maps.


## 🛠 Troubleshooting (macOS + XGBoost)

If you get an error like `libomp.dylib not found`, run:

```bash
brew install libomp
```

If it's already installed but not found by Python:

```bash
ln -s /opt/homebrew/opt/libomp/lib/libomp.dylib /path/to/your/conda/env/lib/libomp.dylib
```

Replace the symlink path as needed. Alternatively, try installing via Conda:

```bash
conda install -c conda-forge libomp
```


## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request with ideas, bug fixes, or improvements.


## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.