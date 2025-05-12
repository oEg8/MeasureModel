"""
posture_predictor_nn.py

This script generates data and trains and evaluates a simple neural network to 
classify posture labels based on 144 input features. 
"""

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


def load_and_preprocess_data(df, test_size=0.2, random_state=42):
    """Load CSV data and preprocess it."""
    df['posture_label'] = df['posture_label'].astype('category').cat.codes

    X = df.drop(columns=['datetime', 'posture_label'])
    y = df['posture_label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train.to_numpy(), dtype=torch.long),
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test.to_numpy(), dtype=torch.long),
        len(np.unique(y))
    )


class SimpleNN(nn.Module):
    """A simple feedforward neural network for multi-class classification."""
    def __init__(self, input_dim, num_classes):
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


def train_model(model, train_loader, epochs=20, lr=0.001, device=None):
    """Train the neural network."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_y.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.4f}")


def evaluate_model(model, X_test_tensor, y_test_tensor, device=None):
    """Evaluate the trained model."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)

    with torch.no_grad():
        logits = model(X_test_tensor)
        y_pred = torch.argmax(logits, dim=1)

    y_true = y_test_tensor.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    print(classification_report(y_true, y_pred, digits=4))


def main():
    """Main function to load data, train model, and evaluate."""

    data = DataGenerator(num_rows=1000, num_features=144, save=False).generate()

    try:
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, num_classes = load_and_preprocess_data(data)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        model = SimpleNN(input_dim=X_train_tensor.shape[1], num_classes=num_classes)
        train_model(model, train_loader, epochs=100, lr=0.001)
        evaluate_model(model, X_test_tensor, y_test_tensor)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
