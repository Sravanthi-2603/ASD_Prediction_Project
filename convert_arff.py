from scipy.io import arff
import pandas as pd

data, meta = arff.loadarff("dataset/autism_screening.arff")

df = pd.DataFrame(data)

# Convert byte values to normal strings
for column in df.select_dtypes([object]).columns:
    df[column] = df[column].str.decode("utf-8")

df.to_csv("dataset/autism_screening.csv", index=False)

print("Dataset converted successfully!")
print(df.head())
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())