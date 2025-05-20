import os
import re
import pandas as pd

input_folder = "metingen/"
output_file = "output.csv"

rows = []

for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(input_folder, filename)
        with open(filepath, "r") as f:
            input_text = f.read()

        weight = None
        height = None
        features = []
        label = None
        correct_heel_pos = None

        for line in input_text.strip().splitlines():
            match = re.match(r"\[(.*?)\] (.*)", line.strip())
            if match:
                _, content = match.groups()
            else:
                content = line.strip()

            if content.startswith("WEIGHT:"):
                weight = float(content.split(":")[1])
            elif content.startswith("HEIGHT:"):
                height = float(content.split(":")[1])
            elif re.match(r"^\d", content):  
                numbers = [float(x.replace(",", ".")) if "," in x else float(x) for x in content.split(",")]
                features.extend(numbers)
            elif content.startswith("LABEL:"):
                label = content.split(":")[1].strip()
            elif content.startswith("CORRECT_HEEL_POS:"):
                correct_heel_pos = content.split(":")[1].strip().lower() == "true"

        row = {
            "weight": weight,
            "height": height,
            "label": label,
            "correct_heel_pos": correct_heel_pos
        }

        for i, val in enumerate(features):
            row[f"feature_{i+1}"] = val

        rows.append(row)

df = pd.DataFrame(rows)
print(df.head())

# df.to_csv(output_file, index=False)
# print(f"CSV-bestand succesvol aangemaakt als: {output_file}")
