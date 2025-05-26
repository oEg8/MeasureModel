import pandas as pd
import ast  

df = pd.read_csv('/Users/hwoutersen/Desktop/School/Jaar3/sem6/MeasureModel/data/new_data.csv')
df.drop(columns=['prediction'], inplace=True)

column_to_split = 'values'

df[column_to_split] = df[column_to_split].apply(ast.literal_eval)

split_df = df[column_to_split].apply(pd.Series)

split_df.columns = [f"feature_{i+1}" for i in range(split_df.shape[1])]

df = pd.concat([df.drop(columns=[column_to_split]), split_df], axis=1)

initial_data = pd.read_csv('data/processed_initial_data.csv')

data = pd.concat([initial_data, df], ignore_index=True)

data.drop_duplicates()


data.to_csv('data/final_combined.csv', index=False)
print(data.shape)
