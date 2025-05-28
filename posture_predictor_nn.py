import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
from safetensors.torch import save_file, load_file
import pickle
import optuna
from optuna.trial import Trial


SCALER_PATH = "scalers/standard_scaler.pkl"


class NeuralNet(nn.Module):
    """
    A feedforward neural network with configurable layers,
    dropout, batch normalization, and LeakyReLU activation.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int] = [128, 64, 32], num_classes: int = 10, dropout_rate: float = 0.5) -> None:
        super(NeuralNet, self).__init__()

        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
            layers.append(nn.Dropout(p=dropout_rate))
            in_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        return self.output_layer(x)


class NNPostureClassifier:
    """
    Encapsulates the workflow for training and evaluating a neural network on posture data.
    """

    def __init__(self, data: pd.DataFrame = pd.DataFrame(), batch_size: int = 16, epochs: int = 20, lr: float = 0.001,
                 hidden_dims: Optional[list[int]] = None, dropout_rate: Optional[float] = None) -> None:
        """
        Initialize the classifier with data and training parameters.

        Args:
            data (pd.DataFrame): Data containing posture features and labels.
            batch_size (int): Number of samples per training batch.
            epochs (int): Number of training epochs.
            lr (float): Learning rate for the optimizer.
        """
        self.data = data
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 64, 32]
        self.dropout_rate = dropout_rate if dropout_rate is not None else 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[NeuralNet] = None
        self.train_loader: Optional[DataLoader] = None
        self.X_test_tensor: Optional[torch.Tensor] = None
        self.y_test_tensor: Optional[torch.Tensor] = None
        self.num_classes: Optional[int] = None


    def preprocess_data(self) -> None:
        """
        Preprocess the input data into PyTorch tensors and create a DataLoader for training.
        """
        df = self.data.copy()
        df['target'] = df['target'].astype('category').cat.codes
        self.label_mapping = {i: label for i, label in enumerate(sorted(self.data['target'].unique()))}

        # delete all rows where all feature columns are zero
        feature_cols = [col for col in df.columns if col.startswith('feature')]
        df = df[~(df[feature_cols].sum(axis=1) == 0)]

        X = df.drop(columns=['time', 'target', 'measurementID'], errors='ignore')
        y = df['target']


        # a pre-trainded scaler is used to scale the data
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=42
        )

        self.num_classes = len(np.unique(y))

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
        self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)


    def build_model(self) -> None:
        """
        Initialize the neural network model.
        """
        input_dim = self.data.shape[1] - 3  # Exclude measurementID, datetime and posture_label
        if self.num_classes is None:
            raise ValueError("Number of classes not set. Run `preprocess_data` first.")
        self.model = NeuralNet(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            num_classes=self.num_classes,
            dropout_rate=self.dropout_rate
        ).to(self.device)


    def train(self) -> None:
        """
        Train the model using the training dataset.
        """
        if self.model is None or self.train_loader is None:
            raise ValueError("Model or data not initialized. Run `preprocess_data` and `build_model` first.")

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in self.train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_y.size(0)

            avg_loss = epoch_loss / len(self.train_loader.dataset)
            print(f"Epoch {epoch + 1}/{self.epochs} - Average Loss: {avg_loss:.4f}")


    def evaluate(self) -> None:
        """
        Evaluate the model on the test dataset.
        """
        if self.model is None:
            raise ValueError("Model not built. Run `build_model` first.")
        if self.X_test_tensor is None or self.y_test_tensor is None:
            raise ValueError("Test data not prepared. Run `preprocess_data` first.")

        self.model.eval()
        self.X_test_tensor = self.X_test_tensor.to(self.device)
        self.y_test_tensor = self.y_test_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(self.X_test_tensor)
            y_pred = torch.argmax(logits, dim=1)

        y_true = self.y_test_tensor.cpu().numpy()
        y_pred = y_pred.cpu().numpy()

        print("\nClassification Report:\n")
        print(classification_report(y_true, y_pred, digits=4))
        print("Confusion Matrix:\n")
        print(confusion_matrix(y_true, y_pred))
        print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")


    def save(self) -> None:
        """
        Saves weights biases & config to a file.
        """
        if not os.path.exists("models"):
            os.makedirs("models", exist_ok=True)

        safetensors_file_path = os.path.join("models", f"nn_posture_model.safetensors")
        config_file_path = os.path.join("models", f"nn_posture_model.json")

        state_dict = self.model.state_dict()
        config = {
            "input_dim": self.data.shape[1] - 3,
            "num_classes": self.num_classes,
            "label_mapping": self.label_mapping,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs
        }

        save_file(state_dict, safetensors_file_path)
        print(f"Saved {type((state_dict))} to: {safetensors_file_path}")

        with open(config_file_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to: {config_file_path}")

    
    def load_model(self, model_name:str) -> None:
        """
        Loads the neural network model's weights & biases from a file.
        """
        safetensors_file_path = os.path.join("models", f"{model_name}.safetensors")
        config_file_path = os.path.join("models", f"{model_name}.json")
        
        state_dict = load_file(safetensors_file_path)
        config = json.load(open(config_file_path))

        self.input_dim = config["input_dim"]
        self.num_classes = config["num_classes"]
        self.label_mapping = config["label_mapping"]
        
        self.model = NeuralNet(input_dim=config["input_dim"], num_classes=config["num_classes"]).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)


    def predict(self, features: list) -> torch.Tensor:
        """
        Predict the posture label for a given set of features.
        """
        logits = self.model.forward(torch.Tensor(features))
        return self.label_mapping[str(int(torch.argmax(logits)))]
    

    def tune_hyperparameters(self, n_trials: int = 30) -> dict:
        """
        Run Optuna hyperparameter tuning and set the best parameters to the model.

        Args:
            n_trials (int): Number of Optuna trials to run.

        Returns:
            dict: Best parameters found.
        """
        study = optuna.create_study(direction="maximize")
        study.optimize(self._optuna_objective, n_trials=n_trials)

        best_params = study.best_trial.params
        print("\nBest trial parameters:")
        for key, value in best_params.items():
            print(f"{key}: {value}")

        self.hidden_dims = [best_params["h1"], best_params["h2"], best_params["h3"]]
        self.dropout_rate = best_params["dropout_rate"]
        self.batch_size = best_params["batch_size"]
        self.lr = best_params["lr"]

        self.preprocess_data()
        self.build_model()
        self.train()
        self.save()

        return best_params
    

    def _optuna_objective(self, trial) -> float:
        """
        Objective function for Optuna to optimize.

        Args:
            trial (optuna.Trial): A single trial.

        Returns:
            float: Accuracy score on test set.
        """
        try:
            # Suggest hyperparameters
            h1 = trial.suggest_int("h1", 32, 256, step=32)
            h2 = trial.suggest_int("h2", 16, 128, step=16)
            h3 = trial.suggest_int("h3", 8, 64, step=8)
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.6, step=0.1)
            batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
            epochs = 100
            lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)

            # Apply parameters
            self.hidden_dims = [h1, h2, h3]
            self.dropout_rate = dropout_rate
            self.batch_size = batch_size
            self.epochs = epochs
            self.lr = lr

            self.preprocess_data()
            self.build_model()
            self.train()

            # Evaluate
            self.model.eval()
            with torch.no_grad():
                logits = self.model(self.X_test_tensor.to(self.device))
                y_pred = torch.argmax(logits, dim=1)

            acc = accuracy_score(self.y_test_tensor.cpu(), y_pred.cpu())
            return acc
        except Exception as e:
            print(f"Trial failed: {e}")
        
        return 0.0


    def run(self) -> None:
        """
        Full pipeline: preprocess data, build model, train and evaluate.
        """
        try:
            print("Preprocessing data...")
            self.preprocess_data()

            print("Building model...")
            self.build_model()

            print("Training model...")
            self.train()

            print("Evaluating model...")
            self.evaluate()

            print("Saving model...")
            self.save()

        except Exception as e:
            print(f"An error occurred during the pipeline: {e}")


def main() -> None:
    data_path = "data/processed/final_combined.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")

    data = pd.read_csv(data_path)

    classifier = NNPostureClassifier(data=data)
    print(classifier.lr, classifier.hidden_dims, classifier.dropout_rate, classifier.batch_size, classifier.epochs)

    use_optuna = True  # Change to True for training with hyperparameter tuning

    if use_optuna:
        print("Start hyperparameter tuning using Optuna...")
        best_params = classifier.tune_hyperparameters(n_trials=30)
        print("Beste parameters found:")
        print(best_params)
    else:
        print("Start standard training...")
        classifier.run()

    print("Done!")


if __name__ == "__main__":
    main()