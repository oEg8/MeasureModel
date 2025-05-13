import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from data_generator import DataGenerator


class SimpleNN(nn.Module):
    """
    A simple feedforward neural network for multi-class classification.
    """

    def __init__(self, input_dim: int, num_classes: int):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.output(x)


class PostureClassifier:
    """
    Encapsulates the workflow for training and evaluating a neural network on posture data.
    """

    def __init__(self, data_df: pd.DataFrame, batch_size: int = 16, epochs: int = 20, lr: float = 0.001):
        """
        Initialize the classifier with data and training parameters.
        """
        self.data_df = data_df
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.train_loader = None
        self.X_test_tensor = None
        self.y_test_tensor = None
        self.num_classes = None


    def preprocess_data(self):
        """
        Preprocess the input data into PyTorch tensors.
        """
        df = self.data_df.copy()
        df['posture_label'] = df['posture_label'].astype('category').cat.codes

        X = df.drop(columns=['datetime', 'posture_label'])
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


    def build_model(self):
        """
        Initialize the neural network model.
        """
        input_dim = self.data_df.shape[1] - 2  # Exclude datetime and label
        self.model = SimpleNN(input_dim=input_dim, num_classes=self.num_classes).to(self.device)


    def train(self):
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


    def evaluate(self):
        """
        Evaluate the model on the test dataset.
        """
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


    def run(self):
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

        except Exception as e:
            print(f"An error occurred during the pipeline: {e}")


def main():
    """
    Example usage of the PostureClassifier.
    """
    generator = DataGenerator(num_rows=1000, num_features=144)
    df = generator.generate()

    classifier = PostureClassifier(data_df=df, epochs=50, lr=0.001)
    classifier.run()


if __name__ == "__main__":
    main()
