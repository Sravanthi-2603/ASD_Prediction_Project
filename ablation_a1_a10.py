
import os
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = os.path.join(
    "dataset",
    "autism_screening.csv"
)

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

print("=" * 80)
print("A1-A10 ABLATION EXPERIMENT")
print("=" * 80)

print("\nDataset shape:", df.shape)


# ============================================================
# 2. CLEAN TARGET
# ============================================================

TARGET = "Class/ASD"

target = (
    df[TARGET]
    .astype(str)
    .str.strip()
    .str.upper()
)

valid_target = target.isin(["NO", "YES"])

df = df.loc[valid_target].copy()

target = target.loc[valid_target]

y = target.map({
    "NO": 0,
    "YES": 1
}).astype(int)

print("\nTarget distribution:")
print(y.value_counts().sort_index())

print("\nNO ASD :", (y == 0).sum())
print("YES ASD:", (y == 1).sum())


# ============================================================
# 3. DEFINE FEATURES
# ============================================================

A1_A10 = [
    "A1_Score",
    "A2_Score",
    "A3_Score",
    "A4_Score",
    "A5_Score",
    "A6_Score",
    "A7_Score",
    "A8_Score",
    "A9_Score",
    "A10_Score",
]

FULL_FEATURES = [
    "A1_Score",
    "A2_Score",
    "A3_Score",
    "A4_Score",
    "A5_Score",
    "A6_Score",
    "A7_Score",
    "A8_Score",
    "A9_Score",
    "A10_Score",
    "age",
    "gender",
    "ethnicity",
    "jundice",
    "austim",
    "contry_of_res",
    "used_app_before",
    "relation",
]


# Check that required columns exist

missing_a1_a10 = [
    col for col in A1_A10
    if col not in df.columns
]

missing_full = [
    col for col in FULL_FEATURES
    if col not in df.columns
]

if missing_a1_a10:
    raise ValueError(
        f"Missing A1-A10 columns: {missing_a1_a10}"
    )

if missing_full:
    raise ValueError(
        f"Missing full-model columns: {missing_full}"
    )


# ============================================================
# 4. CREATE IDENTICAL TRAIN/TEST SPLIT
# ============================================================

indices = np.arange(len(df))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    stratify=y,
    random_state=42
)

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("\nTrain rows:", len(train_idx))
print("Test rows :", len(test_idx))


# ============================================================
# 5. PREPROCESSING FUNCTION
# ============================================================

def create_pipeline(feature_columns):

    X_temp = df[feature_columns]

    numeric_features = X_temp.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = [
        col for col in feature_columns
        if col not in numeric_features
    ]

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    return pipeline


# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, feature_columns):

    print("\n" + "-" * 80)
    print(name)
    print("-" * 80)

    X = df[feature_columns]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    pipeline = create_pipeline(feature_columns)

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    print(f"Number of features: {len(feature_columns)}")
    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"F1 Score           : {f1:.4f}")
    print(f"ROC-AUC            : {roc_auc:.4f}")
    print(f"PR-AUC             : {pr_auc:.4f}")

    return {
        "Experiment": name,
        "Feature_Count": len(feature_columns),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
    }


# ============================================================
# 7. RUN A1-A10 ONLY EXPERIMENT
# ============================================================

a1_a10_result = evaluate_model(
    "A1-A10 ONLY",
    A1_A10
)


# ============================================================
# 8. RUN FULL FEATURE EXPERIMENT
# ============================================================

full_result = evaluate_model(
    "FULL FEATURES WITHOUT result/age_desc",
    FULL_FEATURES
)


# ============================================================
# 9. COMPARE RESULTS
# ============================================================

results = pd.DataFrame([
    a1_a10_result,
    full_result
])

print("\n" + "=" * 80)
print("ABLATION COMPARISON")
print("=" * 80)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 10. CALCULATE DIFFERENCE
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
    "PR-AUC",
]

print("\n" + "=" * 80)
print("FULL MODEL MINUS A1-A10-ONLY")
print("=" * 80)

for metric in metrics:

    difference = (
        full_result[metric]
        - a1_a10_result[metric]
    )

    print(
        f"{metric:<15}: {difference:+.4f}"
    )


# ============================================================
# 11. SAVE RESULTS
# ============================================================

OUTPUT_PATH = "ablation_a1_a10_results.csv"

results.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nResults saved to:")
print(OUTPUT_PATH)

print("\n" + "=" * 80)
print("A1-A10 ABLATION EXPERIMENT COMPLETED")
print("=" * 80)

