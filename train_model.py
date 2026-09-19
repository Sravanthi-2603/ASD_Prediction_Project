import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset/autism_screening.csv")

print("=" * 70)
print("ASD SCREENING - MODEL TRAINING")
print("=" * 70)

print("Dataset shape:", df.shape)


# ============================================================
# 2. CLEAN COLUMN NAMES AND VALUES
# ============================================================

df.columns = df.columns.str.strip()

for column in df.select_dtypes(include=["object"]).columns:
    df[column] = df[column].astype(str).str.strip()


# ============================================================
# 3. TARGET VARIABLE
# ============================================================

target_column = "Class/ASD"

y = df[target_column].map({
    "NO": 0,
    "YES": 1
})

if y.isnull().any():
    print("\nWARNING: Unknown target values detected.")
    print(df.loc[y.isnull(), target_column].value_counts())


X = df.drop(target_column, axis=1)


# ============================================================
# 4. REMOVE REDUNDANT / UNNECESSARY FEATURES
# ============================================================

columns_to_drop = [
    "result",
    "age_desc"
]

X = X.drop(
    columns=columns_to_drop,
    errors="ignore"
)


print("\n" + "=" * 70)
print("FEATURE INFORMATION")
print("=" * 70)

print("Features used for prediction:")
for feature in X.columns:
    print(" -", feature)


print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentage:")
print((y.value_counts(normalize=True) * 100).round(2))


# ============================================================
# 5. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 6. PREPROCESSING
# ============================================================

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


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 8. DEFINE MODELS
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
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}


# ============================================================
# 9. CROSS-VALIDATION FOR MODEL SELECTION
# ============================================================

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


cv_results = {}


for model_name, classifier in models.items():

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            )
        ]
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision"
    }

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    cv_results[model_name] = {

        "CV Accuracy":
            scores["test_accuracy"].mean(),

        "CV Precision":
            scores["test_precision"].mean(),

        "CV Recall/Sensitivity":
            scores["test_recall"].mean(),

        "CV F1":
            scores["test_f1"].mean(),

        "CV ROC-AUC":
            scores["test_roc_auc"].mean(),

        "CV PR-AUC":
            scores["test_pr_auc"].mean(),

        "CV ROC-AUC Std":
            scores["test_roc_auc"].std()
    }


cv_results_df = pd.DataFrame(
    cv_results
).T


print(
    cv_results_df.round(4).to_string()
)


# ============================================================
# 10. SELECT MODEL USING CROSS-VALIDATION
# ============================================================

best_model_name = cv_results_df[
    "CV ROC-AUC"
].idxmax()


print("\n" + "=" * 70)
print("MODEL SELECTED USING CROSS-VALIDATION")
print("=" * 70)

print("Selected model:", best_model_name)

print(
    "Mean CV ROC-AUC:",
    round(
        cv_results_df.loc[
            best_model_name,
            "CV ROC-AUC"
        ],
        4
    )
)


# ============================================================
# 11. TRAIN ALL MODELS ON TRAINING DATA
# ============================================================

trained_models = {}


for model_name, classifier in models.items():

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    trained_models[model_name] = pipeline


# ============================================================
# 12. EVALUATE ALL MODELS ON UNTOUCHED TEST SET
# ============================================================

test_results = {}


for model_name, pipeline in trained_models.items():

    y_pred = pipeline.predict(X_test)

    y_prob = pipeline.predict_proba(
        X_test
    )[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()


    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    pr_auc = average_precision_score(
        y_test,
        y_prob
    )


    test_results[model_name] = {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall/Sensitivity": recall,

        "Specificity": specificity,

        "F1-Score": f1,

        "ROC-AUC": roc_auc,

        "PR-AUC": pr_auc
    }


test_results_df = pd.DataFrame(
    test_results
).T


print("\n" + "=" * 70)
print("FINAL TEST-SET PERFORMANCE")
print("=" * 70)

print(
    test_results_df.round(4).to_string()
)


# ============================================================
# 13. BEST MODEL
# ============================================================

best_model = trained_models[
    best_model_name
]


# ============================================================
# 14. DETAILED TEST EVALUATION
# ============================================================

best_predictions = best_model.predict(
    X_test
)

best_probabilities = best_model.predict_proba(
    X_test
)[:, 1]


print("\n" + "=" * 70)
print("DETAILED EVALUATION")
print("=" * 70)

print("\nSelected model:")
print(best_model_name)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        best_predictions,
        target_names=[
            "NO ASD",
            "ASD"
        ],
        zero_division=0
    )
)


print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        best_predictions
    )
)


# ============================================================
# 15. SAVE TEST SET FOR FUTURE ANALYSIS
# ============================================================

test_data = X_test.copy()

test_data["Actual"] = y_test.values

test_data["Predicted"] = best_predictions

test_data["Probability_ASD"] = best_probabilities


# ============================================================
# 16. SAVE MODEL PACKAGE
# ============================================================

model_package = {

    # Selected model
    "model": best_model,

    "model_name": best_model_name,

    # All trained models
    "all_models": trained_models,

    # Feature information
    "features": list(X.columns),

    "numeric_features":
        numeric_features,

    "categorical_features":
        categorical_features,

    # Cross-validation results
    "cv_results":
        cv_results_df,

    # Final test results
    "test_results":
        test_results_df,

    # Test data
    "X_test":
        X_test,

    "y_test":
        y_test,

    "test_predictions":
        best_predictions,

    "test_probabilities":
        best_probabilities,

    # CV configuration
    "cv_n_splits": 5,

    "random_state": 42
}


joblib.dump(
    model_package,
    "autism_model.pkl"
)


# ============================================================
# 17. SAVE RESULTS AS CSV
# ============================================================

cv_results_df.to_csv(
    "cv_model_comparison.csv"
)

test_results_df.to_csv(
    "test_model_comparison.csv"
)


# ============================================================
# 18. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("Selected model:", best_model_name)

print(
    "Model package saved as:",
    "autism_model.pkl"
)

print(
    "CV results saved as:",
    "cv_model_comparison.csv"
)

print(
    "Test results saved as:",
    "test_model_comparison.csv"
)

print("=" * 70)