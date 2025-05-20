from data_generator import DataGenerator
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

    d = DataGenerator(num_rows=1000, num_features=144)
    data = d.generate(save=True)

    main(data=data, model_type='nn')