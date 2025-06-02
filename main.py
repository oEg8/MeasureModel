from posture_predictor_ml import MLPostureClassifier
from posture_predictor_nn import NNPostureClassifier
import pandas as pd


def main(data: pd.DataFrame, model_type: str):
    """
    Example usage of the PosturePredictor project.
    """
    try:
        if model_type == 'ml':
            evaluator = MLPostureClassifier(data=data)
            evaluator.train_and_evaluate_all()
        elif model_type == 'nn':
            predictor = NNPostureClassifier(data=data)
            predictor.run()
        else:
            raise ValueError("Invalid model type. Choose 'ml' for machine learning or 'nn' for neural networks.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":

    data = pd.read_csv("data/processed/final_combined.csv")

    main(data=data, model_type='nn')