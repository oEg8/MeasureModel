import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
from safetensors.torch import save_file, load_file
import pickle


class SimpleNN(nn.Module):
    """
    A simple feedforward neural network for multi-class classification with optional dropout.
    """
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.output = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return self.output(x)


class NNPostureClassifier:
    """
    Encapsulates the workflow for training and evaluating a neural network on posture data.
    """

    def __init__(self, data: pd.DataFrame = pd.DataFrame(), batch_size: int = 16, epochs: int = 100, lr: float = 0.001, dropout: float = 0.1) -> None:
        """
        Initialize the classifier with data and training parameters.

        Args:
            data (pd.DataFrame): Data containing posture features and labels.
            batch_size (int): Number of samples per training batch.
            epochs (int): Number of training epochs.
            lr (float): Learning rate for the optimizer.
        """
        self.data: pd.DataFrame = data
        self.batch_size: int = batch_size
        self.epochs: int = epochs
        self.lr: float = lr
        self.dropout: float = dropout
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[SimpleNN] = None
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

        with open("scalers/standard_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        X = scaler.transform(X)

        self.input_dim = X.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
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
        if self.num_classes is None:
            raise ValueError("Number of classes not set. Run `preprocess_data` first.")
        self.model = SimpleNN(input_dim=self.input_dim, num_classes=self.num_classes, dropout=self.dropout).to(self.device)


    def train(self) -> None:
        """
        Train the model using the training dataset.
        """
        if self.model is None or self.train_loader is None:
            raise ValueError("Model or data not initialized. Run `preprocess_data` and `build_model` first.")

        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            logits = self.model(self.X_test_tensor)
            loss = criterion(logits, self.y_test_tensor)
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
        print(classification_report(y_true, y_pred, digits=2))
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
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.lr,
        }
    
        save_file(state_dict, safetensors_file_path)
        print(f"Saved {type((state_dict))} to: {safetensors_file_path}")

        with open(config_file_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to: {config_file_path}")

    
    def load_model(self, model_name: str) -> None:
        """
        Loads the neural network model's weights & biases from a file.
        """
        safetensors_file_path = os.path.join("models", f"{model_name}.safetensors")
        config_file_path = os.path.join("models", f"{model_name}.json")

        # Load weights and config
        state_dict = load_file(safetensors_file_path)
        with open(config_file_path, "r") as f:
            config = json.load(f)

        # Load config with fallback defaults
        self.input_dim = config.get("input_dim")
        self.num_classes = config.get("num_classes")
        self.label_mapping = config.get("label_mapping", {})
        self.dropout = config.get("dropout", 0.1)
        self.batch_size = config.get("batch_size", 16)
        self.epochs = config.get("epochs", 100)
        self.lr = config.get("learning_rate", 0.001)

        if self.input_dim is None or self.num_classes is None:
            raise ValueError("Invalid config file: 'input_dim' or 'num_classes' missing.")

        # Build and load model
        self.model = SimpleNN(input_dim=self.input_dim, num_classes=self.num_classes, dropout=self.dropout).to(self.device)
        self.model.load_state_dict(state_dict)


    def predict(self, features: list) -> torch.Tensor:
        """
        Predict the posture label for a given set of features.
        """
        logits = self.model(torch.Tensor(features).to(self.device))
        return self.label_mapping[str(int(torch.argmax(logits)))]


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
    """
    Example usage of the PostureClassifier.
    """
    df = pd.read_csv("data/processed/final_combined.csv")

    classifier = NNPostureClassifier(data=df)
    classifier.run()


if __name__ == "__main__":
    main()