import pandas as pd
import re
import ast
from typing import List, Any

def parse_array(text: Any) -> List[int]:
    """
    Extracts and processes a string of comma-separated integers enclosed in square brackets.
    
    Args:
        text (Any): The input string potentially containing a list-like structure.

    Returns:
        List[int]: A list of exactly 105 integers, padded with 0s if needed.
    """
    if pd.isna(text):
        return [0] * 105

    match = re.search(r'\[(.*?)\]', str(text))
    if match:
        try:
            values = [int(v.strip()) if v.strip().isdigit() else 0 for v in match.group(1).split(',')]
        except Exception as e:
            print(f"Error parsing values: {text} -> {e}")
            return [0] * 105
        return values[:105] + [0] * (105 - len(values))  # Ensure exactly 105 elements

    return [0] * 105


def load_initial_data(filepath: str) -> pd.DataFrame:
    """
    Loads and transforms the initial CSV data into a structured DataFrame.

    Args:
        filepath (str): Path to the initial CSV file.

    Returns:
        pd.DataFrame: Transformed dataset with posture label and 210 features.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise Exception(f"Failed to read file {filepath}: {e}")

    labels = ["correct_posture", "inbalance_left", "inbalance_right", "on_toes", "wrong_foot_position"]
    data = []

    for i in range(0, len(df), 2):
        try:
            obs_name = df.iloc[i, 0]
            if pd.isna(obs_name):
                continue

            for label in labels:
                left_values = parse_array(df.iloc[i][label])
                right_values = parse_array(df.iloc[i + 1][label])
                combined = left_values + right_values

                # Only add rows with active (non-zero) measurements
                if any(x > 0 for x in combined):
                    row = [obs_name] + combined + [label]
                    data.append(row)
        except Exception as e:
            print(f"Skipping rows {i}-{i+1} due to error: {e}")
            continue

    columns = ["observation"] + [f"feature_{i+1}" for i in range(210)] + ["label"]
    return pd.DataFrame(data, columns=columns)


def load_new_data(filepath: str) -> pd.DataFrame:
    """
    Loads and processes new data with flattened feature values.

    Args:
        filepath (str): Path to the new CSV file.

    Returns:
        pd.DataFrame: Flattened dataset with extracted features.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise Exception(f"Failed to read file {filepath}: {e}")

    if 'prediction' in df.columns:
        df.drop(columns=['prediction'], inplace=True)

    column_to_split = 'values'

    try:
        df[column_to_split] = df[column_to_split].apply(ast.literal_eval)
    except Exception as e:
        raise ValueError(f"Error evaluating list values in column '{column_to_split}': {e}")

    split_df = df[column_to_split].apply(pd.Series)
    split_df.columns = [f"feature_{i+1}" for i in range(split_df.shape[1])]
    df = pd.concat([df.drop(columns=[column_to_split]), split_df], axis=1)

    return df


def main():
    """
    Main execution function to merge and export processed initial and new data.
    """
    initial_path = 'data/initial_data.csv'
    new_path = '/Users/hwoutersen/Desktop/School/Jaar3/sem6/MeasureModel/data/new_data.csv'
    output_path = 'data/final_combined.csv'

    # Load and process data
    initial_data = load_initial_data(initial_path)
    new_data = load_new_data(new_path)

    # Combine datasets
    combined_data = pd.concat([initial_data, new_data], ignore_index=True)
    combined_data.drop_duplicates(inplace=True)

    # Save to CSV
    try:
        combined_data.to_csv(output_path, index=False)
        print(f"Final dataset saved to '{output_path}' with shape: {combined_data.shape}")
    except Exception as e:
        print(f"Failed to save final dataset: {e}")


if __name__ == '__main__':
    main()
