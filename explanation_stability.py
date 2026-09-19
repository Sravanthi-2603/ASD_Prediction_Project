import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

import shap


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "dataset/autism_screening.csv"

TARGET = "Class/ASD"

RANDOM_STATE = 42

N_FOLDS = 5

TOP_K = 10


# ============================================================
# FEATURES
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

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("EXPLANATION STABILITY ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(DATA_PATH)

print("\nDataset shape:", data.shape)

data.columns = data.columns.str.strip()

# Clean string values
for column in data.select_dtypes(include=["object", "str"]).columns:
    data[column] = data[column].astype(str).str.strip()


# ============================================================
# TARGET ENCODING
# ============================================================

data[TARGET] = (
    data[TARGET]
    .astype(str)
    .str.upper()
    .map({
        "NO": 0,
        "YES": 1
    })
)

data = data.dropna(subset=[TARGET])

data[TARGET] = data[TARGET].astype(int)


# ============================================================
# PREPARE DATA
# ============================================================

X = data[FEATURES].copy()

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
# MODEL
# ============================================================

model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE
)


# ============================================================
# STRATIFIED K-FOLD
# ============================================================

skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# STORAGE
# ============================================================

fold_importances = []

fold_feature_names = []

fold_top_features = []


# ============================================================
# RUN FOLDS
# ============================================================

print("\n" + "-" * 70)
print("EXPLANATION STABILITY ACROSS FOLDS")
print("-" * 70)

for fold_number, (train_indices, test_indices) in enumerate(
    skf.split(X, y),
    start=1
):

    print(f"\nProcessing Fold {fold_number}...")


    # --------------------------------------------------------
    # SPLIT DATA
    # --------------------------------------------------------

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]


    # --------------------------------------------------------
    # FIT PREPROCESSOR
    # --------------------------------------------------------

    X_train_transformed = preprocessor.fit_transform(
        X_train
    )

    X_test_transformed = preprocessor.transform(
        X_test
    )


    # --------------------------------------------------------
    # GET FEATURE NAMES
    # --------------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )


    # --------------------------------------------------------
    # TRAIN LOGISTIC REGRESSION
    # --------------------------------------------------------

    model.fit(
        X_train_transformed,
        y_train
    )


    # --------------------------------------------------------
    # SHAP EXPLAINER
    # --------------------------------------------------------

    explainer = shap.LinearExplainer(
        model,
        X_train_transformed
    )

    shap_values = explainer(
        X_test_transformed
    )


    # --------------------------------------------------------
    # HANDLE SHAP OUTPUT
    # --------------------------------------------------------

    shap_array = shap_values.values

    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1]


    # --------------------------------------------------------
    # GLOBAL FEATURE IMPORTANCE
    # --------------------------------------------------------

    mean_abs_shap = np.mean(
        np.abs(shap_array),
        axis=0
    )


    # --------------------------------------------------------
    # STORE IMPORTANCE
    # --------------------------------------------------------

    importance_dict = dict(
        zip(
            feature_names,
            mean_abs_shap
        )
    )

    fold_importances.append(
        importance_dict
    )

    fold_feature_names.append(
        feature_names
    )


    # --------------------------------------------------------
    # TOP-K FEATURES
    # --------------------------------------------------------

    sorted_features = sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_features = [
        feature
        for feature, importance
        in sorted_features[:TOP_K]
    ]

    fold_top_features.append(
        top_features
    )


    # --------------------------------------------------------
    # PRINT TOP FEATURES
    # --------------------------------------------------------

    print(f"\nTop {TOP_K} features in Fold {fold_number}:")

    for rank, (feature, importance) in enumerate(
        sorted_features[:TOP_K],
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"{feature:<40} "
            f"{importance:.6f}"
        )


# ============================================================
# COMMON FEATURE SPACE
# ============================================================

all_feature_names = sorted(
    set().union(
        *[
            set(dictionary.keys())
            for dictionary in fold_importances
        ]
    )
)


# ============================================================
# IMPORTANCE MATRIX
# ============================================================

importance_matrix = []

for importance_dict in fold_importances:

    importance_matrix.append([
        importance_dict.get(
            feature,
            0.0
        )
        for feature in all_feature_names
    ])


importance_matrix = np.asarray(
    importance_matrix
)


importance_df = pd.DataFrame(
    importance_matrix,
    columns=all_feature_names
)

importance_df.index = [
    f"Fold_{i}"
    for i in range(1, N_FOLDS + 1)
]


# ============================================================
# MEAN AND STANDARD DEVIATION
# ============================================================

mean_importance = importance_df.mean(
    axis=0
)

std_importance = importance_df.std(
    axis=0
)


stability_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Mean Absolute SHAP": mean_importance.values,
    "SHAP Std": std_importance.values
})


stability_df["Coefficient of Variation"] = (
    stability_df["SHAP Std"] /
    stability_df["Mean Absolute SHAP"].replace(
        0,
        np.nan
    )
)


# ============================================================
# RANK FEATURES
# ============================================================

stability_df = stability_df.sort_values(
    "Mean Absolute SHAP",
    ascending=False
).reset_index(drop=True)

stability_df["Mean Rank"] = (
    stability_df.index + 1
)


# ============================================================
# FEATURE TOP-K FREQUENCY
# ============================================================

top_k_frequency = {}

for feature in all_feature_names:

    count = sum(
        feature in top_features
        for top_features in fold_top_features
    )

    top_k_frequency[feature] = count


stability_df["Top-K Frequency"] = (
    stability_df["Feature"]
    .map(top_k_frequency)
)

stability_df["Top-K Stability"] = (
    stability_df["Top-K Frequency"] /
    N_FOLDS
)


# ============================================================
# JACCARD SIMILARITY
# ============================================================

def jaccard_similarity(set_a, set_b):

    set_a = set(set_a)
    set_b = set(set_b)

    union = set_a.union(set_b)

    if len(union) == 0:
        return 1.0

    intersection = set_a.intersection(set_b)

    return len(intersection) / len(union)


jaccard_values = []


for i in range(N_FOLDS):

    for j in range(i + 1, N_FOLDS):

        similarity = jaccard_similarity(
            fold_top_features[i],
            fold_top_features[j]
        )

        jaccard_values.append({
            "Fold A": i + 1,
            "Fold B": j + 1,
            "Jaccard Similarity": similarity
        })


jaccard_df = pd.DataFrame(
    jaccard_values
)


# ============================================================
# MEAN JACCARD SIMILARITY
# ============================================================

mean_jaccard = (
    jaccard_df["Jaccard Similarity"]
    .mean()
)


# ============================================================
# RANK CORRELATION
# ============================================================

rank_matrix = importance_df.rank(
    axis=1,
    ascending=False,
    method="average"
)

rank_correlations = []


for i in range(N_FOLDS):

    for j in range(i + 1, N_FOLDS):

        correlation = np.corrcoef(
            rank_matrix.iloc[i],
            rank_matrix.iloc[j]
        )[0, 1]

        rank_correlations.append({
            "Fold A": i + 1,
            "Fold B": j + 1,
            "Rank Correlation": correlation
        })


rank_correlation_df = pd.DataFrame(
    rank_correlations
)


mean_rank_correlation = (
    rank_correlation_df["Rank Correlation"]
    .mean()
)


# ============================================================
# PRINT STABILITY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE STABILITY SUMMARY")
print("=" * 70)

print(
    stability_df[
        [
            "Feature",
            "Mean Absolute SHAP",
            "SHAP Std",
            "Coefficient of Variation",
            "Mean Rank",
            "Top-K Frequency",
            "Top-K Stability"
        ]
    ].head(20).to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


print("\n" + "=" * 70)
print("EXPLANATION STABILITY METRICS")
print("=" * 70)

print(
    f"Mean Jaccard Similarity : "
    f"{mean_jaccard:.6f}"
)

print(
    f"Mean Rank Correlation   : "
    f"{mean_rank_correlation:.6f}"
)


# ============================================================
# PRINT FOLD TOP FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP FEATURES BY FOLD")
print("=" * 70)

for i, features in enumerate(
    fold_top_features,
    start=1
):

    print(f"\nFold {i}:")

    for rank, feature in enumerate(
        features,
        start=1
    ):

        print(
            f"{rank:2d}. {feature}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

stability_df.to_csv(
    "explanation_stability_results.csv",
    index=False
)

jaccard_df.to_csv(
    "explanation_jaccard_similarity.csv",
    index=False
)

rank_correlation_df.to_csv(
    "explanation_rank_correlation.csv",
    index=False
)

importance_df.to_csv(
    "explanation_fold_importances.csv"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print("1. explanation_stability_results.csv")
print("2. explanation_jaccard_similarity.csv")
print("3. explanation_rank_correlation.csv")
print("4. explanation_fold_importances.csv")

print("\nExplanation stability analysis completed successfully.")