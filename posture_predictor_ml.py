import pandas as pd
import numpy as np
import pickle

from typing import Tuple
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Constants
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = "data/processed/final_combined.csv"
SCALER_PATH = "scalers/standard_scaler.pkl"
MODEL_PATH = "models/ml_posture_model.pkl"


class MLPostureClassifier:
    """
    A machine learning classifier for posture prediction using RandomForest and GridSearchCV.
    """
    def __init__(self, data: pd.DataFrame, mode: str, random_state: int = RANDOM_STATE) -> None:
        """
        Initializes the classifier with the dataset and model settings.

        Args:
            data (pd.DataFrame): The dataset containing features and labels.
            mode (str): Operation mode ('train' or 'predict').
            random_state (int): Seed for reproducibility.
        """
        self.data = data
        self.mode = mode
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)

        # Load the feature scaler
        try:
            with open(SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Scaler file not found at path: {SCALER_PATH}")

        # Load and split data
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self._prepare_data()


    def _prepare_data(self, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series, pd.Series]:
        """
        Prepares and splits the data into training, validation, and test sets.

        Args:
            test_size (float): Proportion of the dataset to use as test data.

        Returns:
            Tuple containing training, validation, and test features and labels.
        """
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

        df = self.data.dropna().copy()

        if 'target' not in df.columns:
            raise ValueError("Missing 'target' column in dataset.")

        # Encode target labels
        df['target'] = df['target'].astype('category').cat.codes

        # Filter zero-sum feature rows
        feature_cols = [col for col in df.columns if col.startswith("feature")]
        df = df[df[feature_cols].sum(axis=1) > 0]

        # Prepare features and labels
        X = df.drop(columns=['measurementID', 'time', 'target'], errors='ignore')
        y = df['target']

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Split into train, val, test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=self.random_state, stratify=y_temp
        )

        return X_train, X_val, X_test, y_train, y_val, y_test


    def fit(self) -> None:
        """
        Trains the RandomForest model using grid search with cross-validation.
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
            n_jobs=-1,
            verbose=2
        )

        print("Starting GridSearchCV for Random Forest...")
        grid_search.fit(self.X_train, self.y_train)
        print("\nBest parameters found:", grid_search.best_params_)

        self.model = grid_search.best_estimator_


    def evaluate_model(self) -> None:
        """
        Evaluates the model on train, validation, and test sets and prints metrics.
        """
        def _print_evaluation(name: str, y_true: pd.Series, y_pred: np.ndarray) -> None:
            print(f"\n{name} Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
            if name == "Validation":
                print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
                print("Classification Report:\n", classification_report(y_true, y_pred))

        y_pred_train = self.model.predict(self.X_train)
        y_pred_val = self.model.predict(self.X_val)
        y_pred_test = self.model.predict(self.X_test)

        print("Evaluation Results:")
        _print_evaluation("Train", self.y_train, y_pred_train)
        _print_evaluation("Validation", self.y_val, y_pred_val)
        _print_evaluation("Test", self.y_test, y_pred_test)


    def predict(self, features: np.ndarray) -> int:
        """
        Predicts the posture class for a new input.

        Args:
            features (np.ndarray): Input features (1D or 2D array).

        Returns:
            int: Predicted class label.
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        return int(self.model.predict(features_scaled)[0])


    def save_model(self, path: str = MODEL_PATH) -> None:
        """
        Saves the trained model to a file.

        Args:
            path (str): Path to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved successfully to {path}.")


    def load_model(self, path: str = MODEL_PATH) -> None:
        """
        Loads a previously trained model from a file.

        Args:
            path (str): Path to the model file.
        """
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded successfully from {path}.")


def main(mode: str) -> None:
    """
    Main execution function for training or evaluating the posture model.

    Args:
        mode (str): Operation mode - 'train' or 'predict'.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        classifier = MLPostureClassifier(data=df, mode=mode)

        if mode == 'train':
            classifier.fit()
            classifier.evaluate_model()
            classifier.save_model()
        elif mode == 'predict':
            classifier.load_model()
            classifier.evaluate_model()
        else:
            raise ValueError("Mode must be either 'train' or 'predict'.")

    except Exception as e:
        print(f"Error in main(): {e}")


if __name__ == "__main__":
    main("train")
