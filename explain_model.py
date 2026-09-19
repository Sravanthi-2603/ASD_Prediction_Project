import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset/autism_screening.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 2. CLEAN DATA
# ============================================================

df.columns = df.columns.str.strip()

for column in df.select_dtypes(include=["object", "str"]).columns:
    df[column] = df[column].astype(str).str.strip()


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

target_column = "Class/ASD"

y = df[target_column].map({
    "NO": 0,
    "YES": 1
})

X = df.drop(target_column, axis=1)

# Remove redundant fields
X = X.drop(
    columns=["result", "age_desc"],
    errors="ignore"
)


# ============================================================
# 4. IDENTIFY FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()


print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


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


# ============================================================
# 6. XGBOOST MODEL
# ============================================================

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)


# ============================================================
# 7. PREPROCESS DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining XGBoost model...")

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Preprocessing completed!")


# ============================================================
# 8. GET FEATURE NAMES
# ============================================================

feature_names = preprocessor.get_feature_names_out()

# Remove pipeline prefixes for cleaner names
feature_names = [
    name.replace("num__", "")
        .replace("cat__", "")
    for name in feature_names
]

print("\nNumber of transformed features:")
print(len(feature_names))


# ============================================================
# 9. TRAIN XGBOOST
# ============================================================

xgb_model.fit(
    X_train_processed,
    y_train
)

print("XGBoost training completed!")


# ============================================================
# 10. SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP explanations...")

explainer = shap.TreeExplainer(xgb_model)

shap_values = explainer.shap_values(
    X_test_processed
)

print("SHAP calculation completed!")


# ============================================================
# 11. GLOBAL SHAP FEATURE IMPORTANCE
# ============================================================

mean_abs_shap = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": mean_abs_shap
})

importance_df = importance_df.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)

print("\n" + "=" * 60)
print("TOP SHAP FEATURES")
print("=" * 60)

print(
    importance_df.head(15).to_string(index=False)
)


# ============================================================
# 12. SAVE FEATURE IMPORTANCE
# ============================================================

importance_df.to_csv(
    "shap_feature_importance.csv",
    index=False
)

print("\nSHAP feature importance saved as:")
print("shap_feature_importance.csv")


# ============================================================
# 13. SHAP SUMMARY PLOT
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_processed,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_summary_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("SHAP summary plot saved as:")
print("shap_summary_plot.png")


print("\nExplainability analysis completed successfully!")