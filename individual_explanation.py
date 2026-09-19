import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("dataset/autism_screening.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Clean column names and categorical values
# --------------------------------------------------

df.columns = df.columns.str.strip()

categorical_columns = df.select_dtypes(include=["object"]).columns

for col in categorical_columns:
    df[col] = df[col].astype(str).str.strip()


# --------------------------------------------------
# 3. Convert target variable
# --------------------------------------------------

df["Class/ASD"] = df["Class/ASD"].map({
    "NO": 0,
    "YES": 1
})


# --------------------------------------------------
# 4. Remove unnecessary columns
# --------------------------------------------------

# result is highly related to the screening questions.
# age_desc is a redundant description of age.

df = df.drop(columns=["result", "age_desc"])


# --------------------------------------------------
# 5. Separate features and target
# --------------------------------------------------

X = df.drop(columns=["Class/ASD"])
y = df["Class/ASD"]


print("\nFeatures used:")
print(X.columns.tolist())


# --------------------------------------------------
# 6. Identify numerical and categorical features
# --------------------------------------------------

numeric_features = [
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

categorical_features = [
    "gender",
    "ethnicity",
    "jundice",
    "austim",
    "contry_of_res",
    "used_app_before",
    "relation"
]


# --------------------------------------------------
# 7. Preprocessing
# --------------------------------------------------

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
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# --------------------------------------------------
# 8. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 9. Transform training and testing data
# --------------------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# --------------------------------------------------
# 10. Get transformed feature names
# --------------------------------------------------

feature_names = preprocessor.get_feature_names_out()


# --------------------------------------------------
# 11. Train XGBoost model
# --------------------------------------------------

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train_processed, y_train)

print("\nXGBoost model trained successfully!")


# --------------------------------------------------
# 12. Select one individual from test data
# --------------------------------------------------

sample_index = 0

sample = X_test.iloc[[sample_index]]
sample_processed = X_test_processed[sample_index:sample_index + 1]


# --------------------------------------------------
# 13. Make prediction
# --------------------------------------------------

prediction = model.predict(sample_processed)[0]

probability = model.predict_proba(sample_processed)[0][1]


print("\n----------------------------------------")
print("INDIVIDUAL ASD SCREENING RESULT")
print("----------------------------------------")

print("\nInput details:")

print(sample.to_string(index=False))

print("\nPrediction:")

if prediction == 1:
    print("ASD screening prediction: YES")
else:
    print("ASD screening prediction: NO")

print(f"Predicted ASD probability: {probability * 100:.2f}%")


# --------------------------------------------------
# 14. Create SHAP explainer
# --------------------------------------------------

print("\nCalculating SHAP explanation...")

explainer = shap.TreeExplainer(model)

explanation = explainer(sample_processed)


# --------------------------------------------------
# 15. Extract SHAP values
# --------------------------------------------------

shap_values = explanation.values[0]

# Handle possible extra output dimension
if shap_values.ndim > 1:
    shap_values = shap_values[:, 0]


# --------------------------------------------------
# 16. Create explanation table
# --------------------------------------------------

explanation_df = pd.DataFrame({
    "Feature": feature_names,
    "SHAP_Value": shap_values,
    "Absolute_SHAP": np.abs(shap_values)
})

explanation_df = explanation_df.sort_values(
    by="Absolute_SHAP",
    ascending=False
)


# --------------------------------------------------
# 17. Display top contributing features
# --------------------------------------------------

print("\n----------------------------------------")
print("TOP FEATURES CONTRIBUTING TO PREDICTION")
print("----------------------------------------")

print(
    explanation_df[
        ["Feature", "SHAP_Value"]
    ].head(10).to_string(index=False)
)


# --------------------------------------------------
# 18. Explain positive and negative contributors
# --------------------------------------------------

positive_features = explanation_df[
    explanation_df["SHAP_Value"] > 0
].head(5)

negative_features = explanation_df[
    explanation_df["SHAP_Value"] < 0
].sort_values(
    by="SHAP_Value"
).head(5)


print("\nFeatures pushing prediction towards ASD:")

if len(positive_features) == 0:
    print("No strong positive contributors found.")
else:
    print(
        positive_features[
            ["Feature", "SHAP_Value"]
        ].to_string(index=False)
    )


print("\nFeatures pushing prediction away from ASD:")

if len(negative_features) == 0:
    print("No strong negative contributors found.")
else:
    print(
        negative_features[
            ["Feature", "SHAP_Value"]
        ].to_string(index=False)
    )


# --------------------------------------------------
# 19. Save explanation table
# --------------------------------------------------

explanation_df.to_csv(
    "individual_shap_explanation.csv",
    index=False
)


# --------------------------------------------------
# 20. Generate SHAP waterfall plot
# --------------------------------------------------

plt.figure()

shap.plots.waterfall(
    explanation[0],
    show=False
)

plt.tight_layout()

plt.savefig(
    "individual_shap_explanation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\n----------------------------------------")
print("SHAP explanation completed successfully!")
print("----------------------------------------")

print("\nFiles created:")

print("1. individual_shap_explanation.csv")
print("2. individual_shap_explanation.png")