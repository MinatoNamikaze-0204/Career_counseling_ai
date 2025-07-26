# inspect_columns.py
import pandas as pd

df = pd.read_csv('data/AI-based Career Recommendation System.csv')
print("Columns in dataset:")
for idx, col in enumerate(df.columns):
    print(f"{idx+1}. {col}")
