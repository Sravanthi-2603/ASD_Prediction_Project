import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


# ============================================================
# RESPONSE-LEVEL SENSITIVITY / WHAT-IF ANALYSIS
# ============================================================

DATA_PATH = "dataset/autism_screening.csv"

print("=" * 75)
print("RESPONSE-LEVEL SENSITIVITY / WHAT-IF ANALYSIS")
print("=" * 75)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

# Strip whitespace from string columns
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype(str).str.strip()

print("\nDataset shape:", df.shape)


# ============================================================
# 2. DEFINE TARGET
# ============================================================

TARGET = "Class/ASD"

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
# 3. DEFINE FEATURES
# ============================================================

DROP_COLUMNS = [
    TARGET,
    "result",
    "age_desc"
]

X = df.drop(columns=DROP_COLUMNS)
y = df[TARGET]


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


SCREENING_FEATURES = [
    "A1_Score",
    "A2_Score",
    "A3_Score",
    "A4_Score",
    "A5_Score",
    "A6_Score",
    "A7_Score",
    "A8_Score",
    "A9_Score",
    "A10_Score"
]


# ============================================================
# 4. PREPROCESSING
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
        (
            "num",
            numeric_transformer,
            NUMERICAL_FEATURES
        ),
        (
            "cat",
            categorical_transformer,
            CATEGORICAL_FEATURES
        )
    ]
)


# ============================================================
# 5. SAME 80/20 STRATIFIED SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Test samples:", len(X_test))


# ============================================================
# 6. LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


print("\nTraining Logistic Regression...")

pipeline.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ============================================================
# 7. ORIGINAL TEST-SET PROBABILITIES
# ============================================================

original_probabilities = pipeline.predict_proba(
    X_test
)[:, 1]

original_predictions = (
    original_probabilities >= 0.50
).astype(int)


# ============================================================
# 8. FUNCTION FOR ONE WHAT-IF CHANGE
# ============================================================

def calculate_what_if_probability(
    model_pipeline,
    test_data,
    feature_name
):
    """
    Flip a binary A1-A10 response for every test case.

    0 -> 1
    1 -> 0

    Returns the new ASD probabilities.
    """

    modified_data = test_data.copy()

    # Flip the screening response
    modified_data[feature_name] = (
        1 - modified_data[feature_name].astype(int)
    )

    modified_probabilities = model_pipeline.predict_proba(
        modified_data
    )[:, 1]

    return modified_probabilities


# ============================================================
# 9. PER-RESPONSE SENSITIVITY ANALYSIS
# ============================================================

summary_results = []

individual_results = []

print("\n" + "=" * 75)
print("RUNNING WHAT-IF ANALYSIS")
print("=" * 75)


for feature in SCREENING_FEATURES:

    print(f"\nAnalyzing {feature}...")

    # --------------------------------------------------------
    # Original probabilities
    # --------------------------------------------------------

    original_probs = original_probabilities.copy()


    # --------------------------------------------------------
    # Flip the selected response
    # --------------------------------------------------------

    modified_probs = calculate_what_if_probability(
        pipeline,
        X_test,
        feature
    )


    # --------------------------------------------------------
    # Probability change
    # --------------------------------------------------------

    probability_change = (
        modified_probs - original_probs
    )


    absolute_change = np.abs(
        probability_change
    )


    # --------------------------------------------------------
    # Prediction changes
    # --------------------------------------------------------

    original_pred = (
        original_probs >= 0.50
    ).astype(int)

    modified_pred = (
        modified_probs >= 0.50
    ).astype(int)

    prediction_changed = (
        original_pred != modified_pred
    )


    # --------------------------------------------------------
    # Direction of probability change
    # --------------------------------------------------------

    increase_rate = np.mean(
        probability_change > 0
    )

    decrease_rate = np.mean(
        probability_change < 0
    )

    no_change_rate = np.mean(
        probability_change == 0
    )


    # --------------------------------------------------------
    # Mean signed and absolute change
    # --------------------------------------------------------

    mean_signed_change = np.mean(
        probability_change
    )

    mean_absolute_change = np.mean(
        absolute_change
    )

    median_absolute_change = np.median(
        absolute_change
    )

    max_absolute_change = np.max(
        absolute_change
    )


    # --------------------------------------------------------
    # Prediction flip rate
    # --------------------------------------------------------

    prediction_change_rate = np.mean(
        prediction_changed
    )


    # --------------------------------------------------------
    # Store summary
    # --------------------------------------------------------

    summary_results.append({

        "Feature": feature,

        "Mean_Signed_Probability_Change":
            mean_signed_change,

        "Mean_Absolute_Probability_Change":
            mean_absolute_change,

        "Median_Absolute_Probability_Change":
            median_absolute_change,

        "Maximum_Absolute_Probability_Change":
            max_absolute_change,

        "Increase_Rate":
            increase_rate,

        "Decrease_Rate":
            decrease_rate,

        "No_Change_Rate":
            no_change_rate,

        "Prediction_Change_Rate":
            prediction_change_rate,

        "Prediction_Change_Count":
            int(np.sum(prediction_changed))
    })


    # --------------------------------------------------------
    # Store individual test-case results
    # --------------------------------------------------------

    for position, original_index in enumerate(
        X_test.index
    ):

        individual_results.append({

            "Test_Index":
                original_index,

            "Feature":
                feature,

            "Original_Response":
                X_test.loc[
                    original_index,
                    feature
                ],

            "Counterfactual_Response":
                1 - int(
                    X_test.loc[
                        original_index,
                        feature
                    ]
                ),

            "Original_ASD_Probability":
                original_probs[position],

            "Counterfactual_ASD_Probability":
                modified_probs[position],

            "Probability_Change":
                probability_change[position],

            "Absolute_Probability_Change":
                absolute_change[position],

            "Original_Prediction":
                original_pred[position],

            "Counterfactual_Prediction":
                modified_pred[position],

            "Prediction_Changed":
                prediction_changed[position]
        })


# ============================================================
# 10. CREATE SUMMARY DATAFRAME
# ============================================================

summary_df = pd.DataFrame(
    summary_results
)


# ============================================================
# 11. SORT BY MEAN ABSOLUTE SENSITIVITY
# ============================================================

summary_df = summary_df.sort_values(
    by="Mean_Absolute_Probability_Change",
    ascending=False
).reset_index(drop=True)


summary_df["Sensitivity_Rank"] = (
    np.arange(1, len(summary_df) + 1)
)


# ============================================================
# 12. CREATE INDIVIDUAL RESULTS DATAFRAME
# ============================================================

individual_df = pd.DataFrame(
    individual_results
)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

summary_df.to_csv(
    "response_sensitivity_summary.csv",
    index=False
)


individual_df.to_csv(
    "response_sensitivity_individual.csv",
    index=False
)


# ============================================================
# 14. DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("RESPONSE-LEVEL SENSITIVITY SUMMARY")
print("=" * 75)

display_columns = [
    "Sensitivity_Rank",
    "Feature",
    "Mean_Signed_Probability_Change",
    "Mean_Absolute_Probability_Change",
    "Median_Absolute_Probability_Change",
    "Maximum_Absolute_Probability_Change",
    "Increase_Rate",
    "Decrease_Rate",
    "Prediction_Change_Rate",
    "Prediction_Change_Count"
]

print(
    summary_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 15. OVERALL SENSITIVITY STATISTICS
# ============================================================

overall_mean_sensitivity = (
    summary_df[
        "Mean_Absolute_Probability_Change"
    ].mean()
)


overall_prediction_change = (
    summary_df[
        "Prediction_Change_Rate"
    ].mean()
)


most_sensitive_feature = (
    summary_df.iloc[0]["Feature"]
)


least_sensitive_feature = (
    summary_df.iloc[-1]["Feature"]
)


print("\n" + "=" * 75)
print("OVERALL SENSITIVITY STATISTICS")
print("=" * 75)

print(
    f"Mean sensitivity across A1-A10: "
    f"{overall_mean_sensitivity:.4f}"
)

print(
    f"Mean prediction-change rate across A1-A10: "
    f"{overall_prediction_change:.4f}"
)

print(
    f"Highest mean absolute sensitivity: "
    f"{most_sensitive_feature}"
)

print(
    f"Lowest mean absolute sensitivity: "
    f"{least_sensitive_feature}"
)


# ============================================================
# 16. CHECK FOR DIRECTION CONSISTENCY
# ============================================================

print("\n" + "=" * 75)
print("DIRECTION OF PROBABILITY CHANGE")
print("=" * 75)

for _, row in summary_df.iterrows():

    feature = row["Feature"]

    increase = row["Increase_Rate"]
    decrease = row["Decrease_Rate"]
    no_change = row["No_Change_Rate"]

    print(
        f"{feature}: "
        f"Increase={increase:.4f}, "
        f"Decrease={decrease:.4f}, "
        f"No Change={no_change:.4f}"
    )


# ============================================================
# 17. IDENTIFY TEST CASES WITH LARGE CHANGES
# ============================================================

large_change_threshold = 0.10

large_change_cases = individual_df[
    individual_df[
        "Absolute_Probability_Change"
    ] >= large_change_threshold
].copy()


large_change_cases.to_csv(
    "response_sensitivity_large_changes.csv",
    index=False
)


print("\n" + "=" * 75)
print("LARGE PROBABILITY CHANGE ANALYSIS")
print("=" * 75)

print(
    f"Threshold: "
    f"{large_change_threshold:.2f} "
    f"(10 percentage points)"
)

print(
    "Number of feature/test-case combinations "
    f"with >=10 percentage-point change: "
    f"{len(large_change_cases)}"
)


# ============================================================
# 18. FINAL OUTPUT FILES
# ============================================================

print("\n" + "=" * 75)
print("FILES SAVED")
print("=" * 75)

print("1. response_sensitivity_summary.csv")
print("2. response_sensitivity_individual.csv")
print("3. response_sensitivity_large_changes.csv")

print("\nResponse-level sensitivity analysis completed successfully.")