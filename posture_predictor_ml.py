import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from data_generator import DataGenerator

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class PostureModelEvaluator:
    """
    Class for evaluating multiple ML models on posture data.
    """

    def __init__(self, data, test_size=0.2, random_state=RANDOM_STATE):
        """
        Initialises the evaluator with data and model configurations.

        Args:
            data (pd.DataFrame): DataFrame met alle data.
            test_size (float): Verhouding van de testset.
            random_state (int): Seed voor reproduceerbaarheid.
        """
        self.data = data
        self.test_size = test_size
        self.random_state = random_state
        self.models = self._initialize_models()
        self.X_train, self.X_test, self.y_train, self.y_test = self._load_and_prepare_data()


    def _initialize_models(self):
        """
        Initialiseses the ML-models.

        Returns:
            dict: Dictionary with model names and configurations.
        """
        return {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=self.random_state),
            "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs', random_state=self.random_state),
            "Support Vector Machine": SVC(kernel='rbf', probability=True, random_state=self.random_state),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=self.random_state)
        }


    def _load_and_prepare_data(self):
        """
        Loads and prepares the data for training and testing.

        Returns:
            tuple (X_train, X_test, y_train, y_test):
                X_train (np.ndarray): Training features.
                X_test (np.ndarray): Test features.
                y_train (pd.Series): Training labels.
                y_test (pd.Series): Test labels.
        """

        if not isinstance(self.data, pd.DataFrame): 
            raise ValueError("Data must be a pandas DataFrame.")

        df = self.data.copy()

        if 'posture_label' not in df.columns:
            raise ValueError("Column 'posture_label' not found in dataset.")

        df = df.dropna()
        df = df.drop(columns=[col for col in ['Unnamed: 0', 'datetime'] if col in df.columns])

        X = df.drop(columns=['posture_label'])
        y = df['posture_label']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=self.test_size, random_state=self.random_state)

        return X_train, X_test, y_train, y_test


    def train_and_evaluate_all(self):
        """
        Trains and evaluates all models in the evaluator.
        """
        for name, model in self.models.items():
            print(f"\nTraining model: {name}")
            model.fit(self.X_train, self.y_train)
            self._evaluate_model(model, name)


    def _evaluate_model(self, model, name):
        """
        Prints and evaluates the model's performance.

        Args:
            model: The trained ML-model.
            name (str): Name of the model.
        """
        y_pred = model.predict(self.X_test)
        print(f"\nResults for {name}:")
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        print("\nClassification Report:\n", classification_report(self.y_test, y_pred))
        print("Accuracy Score: {:.2f}%".format(accuracy_score(self.y_test, y_pred) * 100))


def main():
    """
    Main function with example usage to generate data and evaluate models.
    """
    generator = DataGenerator(num_rows=1000, num_features=144)
    df = generator.generate(save=True)

    try:
        evaluator = PostureModelEvaluator(data=df)
        evaluator.train_and_evaluate_all()
    except Exception as e:
        print(f"An error occured: {e}")


if __name__ == "__main__":
    main()
