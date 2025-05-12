from datetime import datetime, timedelta
import pandas as pd
import numpy as np


class DataGenerator:
    def __init__(self, num_rows=1000, num_features=144, save=False):
        self.num_rows = num_rows
        self.num_features = num_features
        self.save = save

    def generate(self):
        """Generate random data for testing purposes."""
        start_time = datetime(2001, 12, 9, 17, 0, 0)
        datetime_values = [(start_time + timedelta(days=i)).strftime('%d-%m-%Y %H:%M:%S') for i in range(self.num_rows)]

        posture_labels = ['on toes', 'inbalance left', 'inbalance right', 'correct posture']
        posture_values = np.random.choice(posture_labels, self.num_rows, p=[0.1, 0.1, 0.1, 0.7])

        data = np.random.rand(self.num_rows, self.num_features)

        columns = ['datetime'] + [f'feature_{i}' for i in range(1, self.num_features + 1)] + ['posture_label']

        df = pd.DataFrame(data, columns=columns[1:-1])
        df.insert(0, 'datetime', datetime_values)
        df.insert(self.num_features + 1, 'posture_label', posture_values)

        if self.save:
            df.to_csv('random_data.csv', index=False)

        return df