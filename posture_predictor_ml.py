import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Tuple, Dict, Any
import time
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = "data/final_combined.csv"
SCALER_PATH = "scalers/standard_scaler.pkl"
MODEL_PATH = "models/ml_posture_model.pkl"


class MLPostureClassifier:
    """
    Class for evaluating multiple ML models on posture data.
    """

    def __init__(self, data: pd.DataFrame, mode: str, random_state: int = RANDOM_STATE) -> None:
        """
        Initializes the evaluator with data and model configurations.

        Args:
            data (pd.DataFrame): DataFrame containing all the posture data.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Seed for reproducibility.
        """
        self.data: pd.DataFrame = data
        self.mode = mode
        self.random_state: int = random_state
        self.model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.load_and_prepare_data()

        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)


    def load_and_prepare_data(self, test_size: int = 0.2) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Loads and prepares the data for training and testing.

        Returns:
            Tuple containing:
                X_train (np.ndarray): Training features.
                X_test (np.ndarray): Test features.
                y_train (pd.Series): Training labels.
                y_test (pd.Series): Test labels.
        """
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")

        df = self.data.copy()

        if 'target' not in df.columns:
            raise ValueError("Column 'target' not found in dataset.")

        df = df.dropna()
        df['target'] = df['target'].astype('category').cat.codes

        feature_cols = [col for col in df.columns if col.startswith('feature')]
        df = df[~(df[feature_cols].sum(axis=1) == 0)]

        X = df.drop(columns=['measurementID', 'time', 'target'])
        y = df['target']


        X_scaled = self.scaler.transform(X)

        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=self.random_state, stratify=y)

        # Second split: train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=self.random_state, stratify=y_temp)
        # Result: 60% train, 20% val, 20% test

        return X_train, X_val, X_test, y_train, y_val, y_test


    def fit(self) -> None:
        """
        Trains and evaluates the model using an GridSearchCV for hyperparameter tuning.
        """
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            verbose=2,
            n_jobs=-1
        )

        print("Starting extended Grid Search for Random Forest...")
        grid_search.fit(self.X_train, self.y_train)

        print(f"\nBest parameters found:\n{grid_search.best_params_}")
        self.model = grid_search.best_estimator_


    def evaluate_model(self) -> None:
        """
        Evaluates the trained model on training, validation, and test sets.
        Prints performance metrics.
        """
        y_pred_test = self.model.predict(self.X_test)
        y_pred_val = self.model.predict(self.X_val)
        y_pred_train = self.model.predict(self.X_train)

        print(f"Evaluation results:")
        print("Train Accuracy: {:.2f}%".format(accuracy_score(self.y_train, y_pred_train) * 100))
        print("Validation Accuracy: {:.2f}%".format(accuracy_score(self.y_val, y_pred_val) * 100))
        print("Test Accuracy: {:.2f}%".format(accuracy_score(self.y_test, y_pred_test) * 100))

        print("\nValidation Confusion Matrix:\n", confusion_matrix(self.y_val, y_pred_val))
        print("Validation Classification Report:\n", classification_report(self.y_val, y_pred_val))


    def predict(self, features: np.ndarray) -> int:
        """
        Makes a prediction using the trained model.

        Args:
            features (np.ndarray): A 1D or 2D array of input features.

        Returns:
            int: Predicted class label.
        """

        new_observation = self.scaler.transform(features)

        return int(self.model.predict(new_observation)[0])


    def save_model(self) -> None:
        """
        Saves the trained model to a file.

        Args:
            model_name (str): The name of the model to save.
        """
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved successfully.")


    def load_model(self, path: str = MODEL_PATH) -> None:
        """
        Loads a trained model from a file.

        Args:
            model_name (str): The name of the model to load.
        """
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded successfully.")


def main(mode: str) -> None:
    """
    Main function demonstrating example usage: generate data and evaluate models.


    Args:
        mode (str): The mode of operation, e.g., 'train', 'predict'.
            - train: Train the model.
            - predict: Use the model to make predictions.
    """
    
    df = pd.read_csv(DATA_PATH)
    evaluator = MLPostureClassifier(data=df)

    if mode == 'train':
        try:
            evaluator.fit()
            evaluator.evaluate_model()
            evaluator.save_model()
        except Exception as e:
            print(f"An error occurred: {e}")
    elif mode == 'predict':
        try:
            evaluator.load_model()
            evaluator.evaluate_model()
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        raise ValueError("Invalid mode. Use 'train' or 'predict'.")


if __name__ == "__main__":
    main('train')
