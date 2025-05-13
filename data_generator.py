from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os

RANDOM_STATE = 42


class DataGenerator:
    """
    Class for generating synthetic posture data
    with random features and labels.
    """

    def __init__(self, num_rows=1000, num_features=144, random_state=RANDOM_STATE):
        """
        Initialiseert de generator met aantal rijen en features.

        Args:
            num_rows (int): Aantal te genereren datapunten.
            num_features (int): Aantal kenmerken/features per datapunt.
        """
        self.num_rows = num_rows
        self.num_features = num_features
        self.random_state = random_state
        self.posture_labels = ['on toes', 'inbalance left', 'inbalance right', 'correct posture']
        self.start_time = datetime(2001, 12, 9, 17, 0, 0)
        np.random.seed(RANDOM_STATE)


    def _generate_timestamps(self):
        """
        Generates a list of dates with a one-day difference.

        Returns:
            list: List of date-time strings in the format '%d-%m-%Y %H:%M:%S'.
        """
        return [
            (self.start_time + timedelta(days=i)).strftime('%d-%m-%Y %H:%M:%S')
            for i in range(self.num_rows)
        ]

    def _generate_posture_labels(self):
        """
        Generates random posture labels with specified probabilities.

        Returns:
            np.ndarray: Array with labels.
        """
        return np.random.choice(
            self.posture_labels,
            size=self.num_rows,
            p=[0.1, 0.1, 0.1, 0.7]
        )

    def generate(self, save=False, save_path='random_data.csv'):
        """
        Generates the dataset and returns it as a DataFrame.

        Args:
            save (bool): Whether to save the dataset.
            save_path (str): Path to the file if save=True.

        Returns:
            pd.DataFrame: The generated dataset.
        """
        timestamps = self._generate_timestamps()
        labels = self._generate_posture_labels()
        features = np.random.rand(self.num_rows, self.num_features)

        columns = ['datetime'] + \
                  [f'feature_{i}' for i in range(1, self.num_features + 1)] + \
                  ['posture_label']

        df = pd.DataFrame(features, columns=columns[1:-1])
        df.insert(0, 'datetime', timestamps)
        df.insert(self.num_features + 1, 'posture_label', labels)

        if save:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                df.to_csv(save_path, index=False)
                print(f"Dataset opgeslagen als '{save_path}'")
            except Exception as e:
                print(f"Fout bij opslaan: {e}")

        return df


def main():
    """
    Example usage of the DataGenerator class.
    """
    generator = DataGenerator(num_rows=1000, num_features=144)
    df = generator.generate(save=True)  
    print(df.head())


if __name__ == "__main__":
    main()
