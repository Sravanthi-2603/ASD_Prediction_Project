import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.stats import spearmanr

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "dataset/autism_screening.csv"

df = pd.read_csv(DATA_PATH)

# Remove unnecessary whitespace from column names
df.columns = df.columns.str.strip()

# Strip whitespace from string columns
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype(str).str.strip()


print("=" * 70)
print("MODEL AGREEMENT ANALYSIS")
print("=" * 70)

print("\nDataset shape:", df.shape)


# ============================================================
# 2. DEFINE TARGET
# ============================================================

TARGET = "Class/ASD"

# Convert target to binary
df[TARGET] = (
    df[TARGET]
    .astype(str)
    .str.upper()
    .map({"NO": 0, "YES": 1})
)

print("\nTarget distribution:")
print(df[TARGET].value_counts().sort_index())

print("\n0 = NO ASD")
print("1 = ASD")


# ============================================================
# 3. REMOVE TARGET-RELATED / UNUSED COLUMNS
# ============================================================

# These columns were already excluded in train_model.py
DROP_COLUMNS = [
    TARGET,
    "result",
    "age_desc"
]

X = df.drop(columns=DROP_COLUMNS)
y = df[TARGET]


# ============================================================
# 4. DEFINE NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

NUMERICAL_FEATURES = [
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
    "age"
]

CATEGORICAL_FEATURES = [
    "gender",
    "ethnicity",
    "jundice",
    "austim",
    "contry_of_res",
    "used_app_before",
    "relation"
]


# ============================================================
# 5. PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
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
        ("num", numeric_transformer, NUMERICAL_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES)
    ]
)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTrain samples:", len(X_train))
print("Test samples:", len(X_test))


# ============================================================
# 7. DEFINE FIVE CANDIDATE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),

    "SVM": SVC(
        probability=True,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42
    )
}


# ============================================================
# 8. TRAIN ALL MODELS
# ============================================================

trained_models = {}

prediction_results = {}

probability_results = {}

model_metrics = []


print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)


for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    trained_models[model_name] = pipeline

    # Predictions
    predictions = pipeline.predict(X_test)

    # ASD probability
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    prediction_results[model_name] = predictions
    probability_results[model_name] = probabilities

    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    model_metrics.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "ROC_AUC": roc_auc
    })

    print(f"Accuracy : {accuracy:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 9. MODEL PERFORMANCE SUMMARY
# ============================================================

performance_df = pd.DataFrame(model_metrics)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    performance_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

performance_df.to_csv(
    "model_agreement_performance.csv",
    index=False
)


# ============================================================
# 10. INDIVIDUAL MODEL PREDICTIONS
# ============================================================

individual_predictions = pd.DataFrame({
    "Test_Index": X_test.index,
    "Actual_Class": y_test.values
})

for model_name in models.keys():

    individual_predictions[
        f"{model_name}_Prediction"
    ] = prediction_results[model_name]

    individual_predictions[
        f"{model_name}_ASD_Probability"
    ] = probability_results[model_name]


# ============================================================
# 11. MAJORITY VOTE
# ============================================================

prediction_matrix = np.column_stack(
    [
        prediction_results[model_name]
        for model_name in models.keys()
    ]
)

majority_predictions = (
    np.mean(prediction_matrix, axis=1) >= 0.5
).astype(int)

individual_predictions["Majority_Prediction"] = majority_predictions

individual_predictions["Majority_Agreement_Count"] = np.maximum(
    prediction_matrix.sum(axis=1),
    prediction_matrix.shape[1] - prediction_matrix.sum(axis=1)
)

individual_predictions["Majority_Agreement_Rate"] = (
    individual_predictions["Majority_Agreement_Count"] /
    len(models)
)


# ============================================================
# 12. DISAGREEMENT ANALYSIS
# ============================================================

individual_predictions["Model_Disagreement"] = (
    prediction_matrix.max(axis=1)
    != prediction_matrix.min(axis=1)
)


individual_predictions["Agreement_Category"] = np.where(
    individual_predictions["Majority_Agreement_Rate"] == 1.0,
    "Complete Agreement",
    np.where(
        individual_predictions["Majority_Agreement_Rate"] >= 0.8,
        "High Agreement",
        "Disagreement"
    )
)


# ============================================================
# 13. MODEL PROBABILITY STANDARD DEVIATION
# ============================================================

probability_matrix = np.column_stack(
    [
        probability_results[model_name]
        for model_name in models.keys()
    ]
)

individual_predictions["Probability_Mean"] = (
    probability_matrix.mean(axis=1)
)

individual_predictions["Probability_Std"] = (
    probability_matrix.std(axis=1)
)


# ============================================================
# 14. PAIRWISE PREDICTION AGREEMENT
# ============================================================

model_names = list(models.keys())

pairwise_prediction_agreement = []


for i in range(len(model_names)):

    for j in range(i + 1, len(model_names)):

        model_a = model_names[i]
        model_b = model_names[j]

        pred_a = prediction_results[model_a]
        pred_b = prediction_results[model_b]

        agreement = np.mean(pred_a == pred_b)

        pairwise_prediction_agreement.append({
            "Model_A": model_a,
            "Model_B": model_b,
            "Prediction_Agreement": agreement,
            "Prediction_Disagreement": 1 - agreement
        })


pairwise_prediction_df = pd.DataFrame(
    pairwise_prediction_agreement
)


# ============================================================
# 15. PAIRWISE PROBABILITY CORRELATION
# ============================================================

pairwise_probability_correlation = []


for i in range(len(model_names)):

    for j in range(i + 1, len(model_names)):

        model_a = model_names[i]
        model_b = model_names[j]

        prob_a = probability_results[model_a]
        prob_b = probability_results[model_b]

        pearson_corr = np.corrcoef(
            prob_a,
            prob_b
        )[0, 1]

        spearman_corr, _ = spearmanr(
            prob_a,
            prob_b
        )

        pairwise_probability_correlation.append({
            "Model_A": model_a,
            "Model_B": model_b,
            "Pearson_Probability_Correlation": pearson_corr,
            "Spearman_Probability_Correlation": spearman_corr
        })


pairwise_probability_df = pd.DataFrame(
    pairwise_probability_correlation
)


# ============================================================
# 16. OVERALL AGREEMENT STATISTICS
# ============================================================

complete_agreement_rate = np.mean(
    individual_predictions["Majority_Agreement_Rate"] == 1.0
)

high_agreement_rate = np.mean(
    individual_predictions["Majority_Agreement_Rate"] >= 0.8
)

disagreement_rate = np.mean(
    individual_predictions["Model_Disagreement"]
)


mean_agreement_rate = (
    individual_predictions["Majority_Agreement_Rate"]
    .mean()
)

mean_probability_std = (
    individual_predictions["Probability_Std"]
    .mean()
)


overall_agreement = pd.DataFrame({

    "Metric": [
        "Mean Majority Agreement Rate",
        "Complete Agreement Rate",
        "High Agreement Rate",
        "Model Disagreement Rate",
        "Mean Probability Standard Deviation"
    ],

    "Value": [
        mean_agreement_rate,
        complete_agreement_rate,
        high_agreement_rate,
        disagreement_rate,
        mean_probability_std
    ]
})


# ============================================================
# 17. SAVE RESULTS
# ============================================================

individual_predictions.to_csv(
    "model_agreement_individual_predictions.csv",
    index=False
)

pairwise_prediction_df.to_csv(
    "model_agreement_pairwise_predictions.csv",
    index=False
)

pairwise_probability_df.to_csv(
    "model_agreement_probability_correlation.csv",
    index=False
)

overall_agreement.to_csv(
    "model_agreement_summary.csv",
    index=False
)


# ============================================================
# 18. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PAIRWISE PREDICTION AGREEMENT")
print("=" * 70)

print(
    pairwise_prediction_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n" + "=" * 70)
print("PAIRWISE PROBABILITY CORRELATION")
print("=" * 70)

print(
    pairwise_probability_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n" + "=" * 70)
print("OVERALL MODEL AGREEMENT")
print("=" * 70)

print(
    overall_agreement.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


print("\n" + "=" * 70)
print("PREDICTION SET DISTRIBUTION")
print("=" * 70)

print(
    individual_predictions[
        "Agreement_Category"
    ].value_counts()
)


print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print("1. model_agreement_performance.csv")
print("2. model_agreement_individual_predictions.csv")
print("3. model_agreement_pairwise_predictions.csv")
print("4. model_agreement_probability_correlation.csv")
print("5. model_agreement_summary.csv")

print("\nModel agreement analysis completed successfully.")