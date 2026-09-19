import pandas as pd

# Load dataset
df = pd.read_csv("dataset/autism_screening.csv")

print("=" * 60)
print("AUTISM SCREENING DATASET ANALYSIS")
print("=" * 60)

# Basic information
print("\n1. DATASET SHAPE")
print(df.shape)

print("\n2. COLUMN NAMES")
print(df.columns.tolist())

# Target distribution
print("\n3. ASD TARGET DISTRIBUTION")
print(df["Class/ASD"].value_counts())

print("\nTarget percentages:")
print(
    df["Class/ASD"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# Calculate screening score
score_columns = [
    f"A{i}_Score"
    for i in range(1, 11)
]

df["Total_Score"] = df[score_columns].sum(axis=1)

# Score distribution
print("\n4. TOTAL SCREENING SCORE DISTRIBUTION")
print(df["Total_Score"].value_counts().sort_index())

# Score versus target
print("\n5. SCREENING SCORE VS ASD RESULT")

score_target = (
    df.groupby("Total_Score")["Class/ASD"]
    .value_counts()
    .unstack(fill_value=0)
)

print(score_target)

# Specifically score 5
print("\n6. RECORDS WITH SCREENING SCORE = 5")

score_5 = df[df["Total_Score"] == 5]

print(
    score_5[
        score_columns +
        [
            "age",
            "gender",
            "ethnicity",
            "jundice",
            "austim",
            "contry_of_res",
            "used_app_before",
            "relation",
            "Class/ASD"
        ]
    ].to_string(index=False)
)

# Missing values
print("\n7. MISSING VALUES")
print(df.isnull().sum())

# Question mark values
print("\n8. '?' VALUES")
print((df == "?").sum())

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)