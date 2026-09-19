import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "dataset/autism_screening.csv"
RANDOM_STATE = 42

TARGET = "Class/ASD"

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
# HELPER FUNCTIONS
# ============================================================

def conformal_quantile(scores, alpha):
    """
    Calculate the finite-sample split-conformal quantile.
    """
    scores = np.sort(np.asarray(scores))
    n = len(scores)

    index = int(np.ceil((n + 1) * (1 - alpha))) - 1

    index = max(0, min(index, n - 1))

    return scores[index]


def create_prediction_set(probability_asd, threshold):
    """
    Create a binary conformal prediction set.

    Class 0 = NO ASD
    Class 1 = ASD
    """

    probability_no_asd = 1.0 - probability_asd

    prediction_set = []

    # Nonconformity score for class 0
    score_class_0 = 1.0 - probability_no_asd

    # Nonconformity score for class 1
    score_class_1 = 1.0 - probability_asd

    if score_class_0 <= threshold:
        prediction_set.append(0)

    if score_class_1 <= threshold:
        prediction_set.append(1)

    return prediction_set


def prediction_set_label(prediction_set):
    """
    Convert prediction set to a readable label.
    """

    if prediction_set == [0]:
        return "{0}"

    if prediction_set == [1]:
        return "{1}"

    if prediction_set == [0, 1]:
        return "{0,1}"

    return "{}"


def is_covered(actual_class, prediction_set):
    """
    Check whether the true class is contained
    in the conformal prediction set.
    """

    return int(actual_class in prediction_set)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CONFORMAL PREDICTION - INDEPENDENT UNCERTAINTY ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

data = pd.read_csv(DATA_PATH)

print("\nDataset shape:", data.shape)

# Remove leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Clean string columns
for column in data.select_dtypes(include=["object", "str"]).columns:
    data[column] = data[column].astype(str).str.strip()

# Convert target
data[TARGET] = (
    data[TARGET]
    .astype(str)
    .str.upper()
    .map({
        "NO": 0,
        "YES": 1
    })
)

# Remove rows with missing target
data = data.dropna(subset=[TARGET])

data[TARGET] = data[TARGET].astype(int)

print("\nTarget distribution:")
print(data[TARGET].value_counts().sort_index())


# ============================================================
# PREPARE FEATURES
# ============================================================

X = data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
y = data[TARGET].copy()


# ============================================================
# PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, NUMERICAL_FEATURES),
    ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
])


# ============================================================
# LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# ============================================================
# ORIGINAL TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\n" + "-" * 70)
print("DATA SPLIT")
print("-" * 70)

print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")


# ============================================================
# CONFORMAL CALIBRATION SPLIT
# ============================================================

X_model_train, X_calibration, y_model_train, y_calibration = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    stratify=y_train,
    random_state=RANDOM_STATE
)

print("\n" + "-" * 70)
print("CONFORMAL DATA SPLIT")
print("-" * 70)

print(f"Model-training samples        : {len(X_model_train)}")
print(f"Conformal calibration samples : {len(X_calibration)}")
print(f"Final untouched test samples  : {len(X_test)}")


# ============================================================
# TRAIN MODEL
# ============================================================

pipeline.fit(X_model_train, y_model_train)


# ============================================================
# CALIBRATION NONCONFORMITY SCORES
# ============================================================

calibration_probabilities = pipeline.predict_proba(
    X_calibration
)

calibration_scores = []

for i, actual_class in enumerate(y_calibration):

    probability_true_class = calibration_probabilities[
        i,
        int(actual_class)
    ]

    score = 1.0 - probability_true_class

    calibration_scores.append(score)

calibration_scores = np.asarray(calibration_scores)


# ============================================================
# NONCONFORMITY SCORE SUMMARY
# ============================================================

print("\n" + "-" * 70)
print("NONCONFORMITY SCORES")
print("-" * 70)

print(f"Minimum : {calibration_scores.min():.6f}")
print(f"Mean    : {calibration_scores.mean():.6f}")
print(f"Median  : {np.median(calibration_scores):.6f}")
print(f"Maximum : {calibration_scores.max():.6f}")


# ============================================================
# TEST PROBABILITIES
# ============================================================

test_probabilities = pipeline.predict_proba(X_test)

test_asd_probabilities = test_probabilities[:, 1]

test_actual_classes = np.asarray(y_test)


# ============================================================
# CONFORMAL ANALYSIS
# ============================================================

target_coverages = [0.80, 0.90, 0.95]

results = []

individual_results_90 = None


for target_coverage in target_coverages:

    alpha = 1.0 - target_coverage

    threshold = conformal_quantile(
        calibration_scores,
        alpha
    )

    prediction_sets = []

    covered_values = []

    for probability_asd, actual_class in zip(
        test_asd_probabilities,
        test_actual_classes
    ):

        prediction_set = create_prediction_set(
            probability_asd,
            threshold
        )

        prediction_sets.append(prediction_set)

        covered = is_covered(
            actual_class,
            prediction_set
        )

        covered_values.append(covered)

    covered_values = np.asarray(covered_values)

    # --------------------------------------------------------
    # BASIC METRICS
    # --------------------------------------------------------

    empirical_coverage = covered_values.mean()

    set_sizes = np.asarray([
        len(prediction_set)
        for prediction_set in prediction_sets
    ])

    average_set_size = set_sizes.mean()

    singleton_rate = np.mean(set_sizes == 1)

    ambiguous_rate = np.mean(set_sizes == 2)

    empty_rate = np.mean(set_sizes == 0)

    coverage_gap = (
        target_coverage -
        empirical_coverage
    )


    # --------------------------------------------------------
    # CLASS-WISE COVERAGE
    # --------------------------------------------------------

    no_asd_mask = test_actual_classes == 0
    asd_mask = test_actual_classes == 1

    if no_asd_mask.sum() > 0:
        no_asd_coverage = covered_values[
            no_asd_mask
        ].mean()
    else:
        no_asd_coverage = np.nan

    if asd_mask.sum() > 0:
        asd_coverage = covered_values[
            asd_mask
        ].mean()
    else:
        asd_coverage = np.nan


    # --------------------------------------------------------
    # PREDICTION-SET DISTRIBUTION
    # --------------------------------------------------------

    set_labels = [
        prediction_set_label(prediction_set)
        for prediction_set in prediction_sets
    ]

    count_0 = set_labels.count("{0}")
    count_1 = set_labels.count("{1}")
    count_01 = set_labels.count("{0,1}")
    count_empty = set_labels.count("{}")


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append({
        "Target Coverage": target_coverage,
        "Alpha": alpha,
        "Conformal Threshold": threshold,
        "Empirical Coverage": empirical_coverage,
        "Coverage Gap": coverage_gap,
        "NO-ASD Coverage": no_asd_coverage,
        "ASD Coverage": asd_coverage,
        "Average Set Size": average_set_size,
        "Singleton Rate": singleton_rate,
        "Ambiguous Rate": ambiguous_rate,
        "Empty Set Rate": empty_rate,
        "Set {0} Count": count_0,
        "Set {1} Count": count_1,
        "Set {0,1} Count": count_01,
        "Set {} Count": count_empty
    })


    # --------------------------------------------------------
    # SAVE INDIVIDUAL RESULTS FOR 90% LEVEL
    # --------------------------------------------------------

    if target_coverage == 0.90:

        individual_rows = []

        for i, (
            probability_asd,
            actual_class,
            prediction_set
        ) in enumerate(zip(
            test_asd_probabilities,
            test_actual_classes,
            prediction_sets
        )):

            set_size = len(prediction_set)

            if set_size == 1:
                uncertainty_indicator = "Low"

            elif set_size == 2:
                uncertainty_indicator = "Higher"

            else:
                uncertainty_indicator = "Undefined"

            individual_rows.append({
                "Test Index": i,
                "Actual Class": int(actual_class),
                "ASD Probability": probability_asd,
                "Prediction Set": prediction_set_label(
                    prediction_set
                ),
                "Prediction Set Size": set_size,
                "Covered": int(
                    actual_class in prediction_set
                ),
                "Uncertainty Indicator":
                    uncertainty_indicator
            })

        individual_results_90 = pd.DataFrame(
            individual_rows
        )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CONFORMAL PREDICTION RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# PREDICTION-SET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION-SET DISTRIBUTION")
print("=" * 70)

for _, row in results_df.iterrows():

    print(
        f"\nTarget Coverage: "
        f"{row['Target Coverage']:.0%}"
    )

    print(
        f"  {{0}}   : "
        f"{int(row['Set {0} Count'])}"
    )

    print(
        f"  {{1}}   : "
        f"{int(row['Set {1} Count'])}"
    )

    print(
        f"  {{0,1}} : "
        f"{int(row['Set {0,1} Count'])}"
    )

    print(
        f"  {{}}    : "
        f"{int(row['Set {} Count'])}"
    )


# ============================================================
# 90% INDIVIDUAL PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("INDIVIDUAL CONFORMAL PREDICTIONS - FIRST 10")
print("=" * 70)

print(
    individual_results_90.head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# SAVE FILES
# ============================================================

results_df.to_csv(
    "conformal_prediction_results.csv",
    index=False
)

individual_results_90.to_csv(
    "conformal_individual_predictions.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print("1. conformal_prediction_results.csv")
print("2. conformal_individual_predictions.csv")

print("\nConformal prediction analysis completed successfully.")