import pandas as pd
import re

# Bestand inlezen
filename = 'data/initial_data.csv'
df = pd.read_csv(filename)

# De kolommen met labels
labels = ["correct_posture", "inbalance_left", "inbalance_right", "on_toes", "wrong_foot_position"]

# Functie om drukpunten uit een string te halen
def parse_array(text):
    if pd.isna(text):
        return [0] * 105
    match = re.search(r'\[(.*?)\]', str(text))
    if match:
        values = [int(v.strip()) if v.strip().isdigit() else 0 for v in match.group(1).split(',')]
        return values[:105] + [0] * (105 - len(values))  # vul aan tot 105
    return [0] * 105

data = []

for i in range(0, len(df), 2):
    obs_name = df.iloc[i, 0]
    if pd.isna(obs_name):
        continue

    for label in labels:
        left_values = parse_array(df.iloc[i][label])
        right_values = parse_array(df.iloc[i + 1][label])
        combined = left_values + right_values

        # Als er enige waarde > 0 is, dan is deze meting actief
        if any(x > 0 for x in combined):
            row = [obs_name] + combined + [label]
            data.append(row)

# DataFrame bouwen
column_names = ["observation"] + [f"feature_{i+1}" for i in range(210)] + ["label"]
result_df = pd.DataFrame(data, columns=column_names)

# Resultaat bekijken
print(f"Aantal rijen: {len(result_df)} | Aantal kolommen: {len(result_df.columns)}")

# Optioneel opslaan
result_df.to_csv("data/processed_output.csv", index=False)
