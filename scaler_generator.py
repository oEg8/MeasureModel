import pandas as pd
from sklearn.preprocessing import StandardScaler
from pickle import dump

def generate_dummy_data() -> pd.DataFrame:
    """
    Generates dummy data for testing purposes.
    
    Returns:
        pd.DataFrame: DataFrame with dummy posture data.
    """
    # Create a DataFrame with random values between 0 and 1024

    cols = [f"feature_{i+1}" for i in range(210)]
    data = [0 for _ in range(210)]
    data2 = [1024 for _ in range(210)]
    
    df = pd.DataFrame([data, data2], columns=cols)
    
    return df


def generate_scaler(data_path: str, scaler_path: str) -> None:
    """
    Generates a StandardScaler for the dataset and saves it to a file.
    
    Args:
        data_path (str): Path to the CSV file containing the posture data.
        scaler_path (str): Path where the scaler will be saved.
    """
    # Load the data
    df = pd.read_csv(data_path)
        
    # Create a StandardScaler instance
    scaler = StandardScaler()
    
    # Fit the scaler on the feature data
    scaler.fit(df)

    # Save the scaler to a file
    with open(scaler_path, 'wb') as f:
        dump(scaler, f)