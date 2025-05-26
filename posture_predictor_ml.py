import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data.data_generator import DataGenerator
from typing import Tuple, Dict, Any
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class MLPostureClassifier:
    """
    Class for evaluating multiple ML models on posture data.
    """

    def __init__(self, data: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE) -> None:
        """
        Initializes the evaluator with data and model configurations.

        Args:
            data (pd.DataFrame): DataFrame containing all the posture data.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Seed for reproducibility.
        """
        self.data: pd.DataFrame = data
        self.test_size: float = test_size
        self.random_state: int = random_state
        self.models: Dict[str, Any] = self._initialize_models()
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self._load_and_prepare_data()


    def _initialize_models(self) -> Dict[str, Any]:
        """
        Initializes the machine learning models.

        Returns:
            dict: Dictionary mapping model names to instantiated model objects.
        """
        return {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=self.random_state),
            "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=self.random_state),
            "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=self.random_state),
            "Support Vector Machine (RBF)": SVC(kernel='rbf', probability=True, random_state=self.random_state),
            "Support Vector Machine (Linear)": LinearSVC(max_iter=10000, random_state=self.random_state),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
            "CatBoost": CatBoostClassifier(verbose=0, random_state=self.random_state),
        }


    def _load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
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

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=self.test_size, random_state=self.random_state, stratify=y)

        # Second split: train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=self.random_state, stratify=y_temp)
        # Result: 60% train, 20% val, 20% test

        return X_train, X_val, X_test, y_train, y_val, y_test


    def train_and_evaluate_all(self) -> None:
        """
        Trains and evaluates all configured models.
        """
        for name, model in self.models.items():
            start_time = time.time()
            print(f"\nTraining model: {name}")
            model.fit(self.X_train, self.y_train)
            self._evaluate_model(model, name)
            print(f"Time taken for {name}: {time.time() - start_time:.2f} seconds\n")


    def train_and_evaluate_model(self, model_name: str) -> None:
        """
        Trains and evaluates a single configured model.
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found in the available models.")
        start_time = time.time()
        print(f"\nTraining model: {model_name}")
        model.fit(self.X_train, self.y_train)
        self._evaluate_model(model, model_name)
        print(f"Time taken for {model_name}: {time.time() - start_time:.2f} seconds\n")


    def _evaluate_model(self, model: Any, name: str) -> None:
        """
        Evaluates a single model and prints performance metrics.

        Args:
            model: The trained machine learning model.
            name (str): The name of the model.
        """
        y_pred_test = model.predict(self.X_test)
        y_pred_val = model.predict(self.X_val)
        y_pred_train = model.predict(self.X_train)

        print(f"Results for {name}:")

        print("Train Accuracy: {:.2f}%".format(accuracy_score(self.y_train, y_pred_train) * 100))
        print("Validation Accuracy: {:.2f}%".format(accuracy_score(self.y_val, y_pred_val) * 100))
        print("Test Accuracy: {:.2f}%".format(accuracy_score(self.y_test, y_pred_test) * 100))

        print("\nValidation Confusion Matrix:\n", confusion_matrix(self.y_val, y_pred_val))
        print("Validation Classification Report:\n", classification_report(self.y_val, y_pred_val))


def main() -> None:
    """
    Main function demonstrating example usage: generate data and evaluate models.
    """
    df = pd.read_csv('data/final_combined.csv')

    try:
        evaluator = MLPostureClassifier(data=df)
        evaluator.train_and_evaluate_all()
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
