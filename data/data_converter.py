import pandas as pd
import re
import ast

### converter voor initiele data ###

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

initial_data = pd.DataFrame(data, columns=column_names)




### converter voor nieuwe data ###


df = pd.read_csv('/Users/hwoutersen/Desktop/School/Jaar3/sem6/MeasureModel/data/new_data.csv')
df.drop(columns=['prediction'], inplace=True)

column_to_split = 'values'

df[column_to_split] = df[column_to_split].apply(ast.literal_eval)

split_df = df[column_to_split].apply(pd.Series)

split_df.columns = [f"feature_{i+1}" for i in range(split_df.shape[1])]

df = pd.concat([df.drop(columns=[column_to_split]), split_df], axis=1)

data = pd.concat([initial_data, df], ignore_index=True)

data.drop_duplicates()

data.to_csv('data/final_combined.csv', index=False)

print(data.shape)
