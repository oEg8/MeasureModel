import os
from datetime import datetime
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
from data_generator import DataGenerator
from safetensors.torch import save_file, load_file


class SimpleNN(nn.Module):
    """
    A simple feedforward neural network for multi-class classification.
    """

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.output(x)


class NNPostureClassifier:
    """
    Encapsulates the workflow for training and evaluating a neural network on posture data.
    """

    def __init__(self, data: pd.DataFrame = pd.DataFrame(), batch_size: int = 16, epochs: int = 20, lr: float = 0.001) -> None:
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
        df['posture_label'] = df['posture_label'].astype('category').cat.codes
        self.label_mapping = {i: label for i, label in enumerate(sorted(self.data['posture_label'].unique()))}

        X = df.drop(columns=['datetime', 'posture_label'], errors='ignore')
        y = df['posture_label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.num_classes = len(np.unique(y))

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
        self.X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)


    def build_model(self) -> None:
        """
        Initialize the neural network model.
        """
        input_dim = self.data.shape[1] - 2  # Exclude datetime and posture_label
        if self.num_classes is None:
            raise ValueError("Number of classes not set. Run `preprocess_data` first.")
        self.model = SimpleNN(input_dim=input_dim, num_classes=self.num_classes).to(self.device)


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


    def save(self) -> None:
        """
        Saves weights biases & config to a file.
        """
        if not os.path.exists("model"):
            os.makedirs("model", exist_ok=True)

        now = datetime.now().strftime("%Y-%m-%d_%H-%M")
        safetensors_file_path = os.path.join("model", f"posture_model_at_{now}.safetensors")
        config_file_path = os.path.join("model", f"posture_model_at_{now}.json")

        state_dict = self.model.state_dict()
        config = {
            "input_dim" : self.data.shape[1] - 2,  # Exclude datetime and posture_label
            "num_classes" : self.num_classes, 
            "label_mapping" : self.label_mapping
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
        safetensors_file_path = os.path.join("model", f"{model_name}.safetensors")
        config_file_path = os.path.join("model", f"{model_name}.json")
        
        state_dict = load_file(safetensors_file_path)
        config = json.load(open(config_file_path))

        self.input_dim = config["input_dim"]
        self.num_classes = config["num_classes"]
        self.label_mapping = config["label_mapping"]
        
        self.model = SimpleNN(input_dim=config["input_dim"], num_classes=config["num_classes"]).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    def predict(self, features: list) -> torch.Tensor:
        logits = self.model.forward(torch.Tensor(features))
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
    generator = DataGenerator(num_rows=1000, num_features=144)
    df = generator.generate()

    classifier = NNPostureClassifier(data=df, epochs=50, lr=0.001)
    classifier.run()


if __name__ == "__main__":
    main()
