import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve
)
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    log_loss
)
from sklearn.base import clone


# ============================================================
# 1. LOAD TRAINING PACKAGE
# ============================================================

package = joblib.load("autism_model.pkl")

best_model = package["model"]
best_model_name = package["model_name"]

X_test = package["X_test"]
y_test = package["y_test"]

print("=" * 70)
print("PROBABILITY RELIABILITY / CALIBRATION ANALYSIS")
print("=" * 70)

print("Selected model:", best_model_name)

print("Test samples:", len(X_test))


# ============================================================
# 2. LOAD ORIGINAL DATASET
# ============================================================

df = pd.read_csv(
    "dataset/autism_screening.csv"
)

df.columns = df.columns.str.strip()

for column in df.select_dtypes(
    include=["object"]
).columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
    )


# ============================================================
# 3. PREPARE TARGET
# ============================================================

target_column = "Class/ASD"

y = df[target_column].map({
    "NO": 0,
    "YES": 1
})

X = df.drop(
    target_column,
    axis=1
)


# ============================================================
# 4. REMOVE REDUNDANT FEATURES
# ============================================================

X = X.drop(
    columns=[
        "result",
        "age_desc"
    ],
    errors="ignore"
)


# ============================================================
# 5. RECREATE SAME TRAIN / TEST SPLIT
# ============================================================

X_train, X_test_original, y_train, y_test_original = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 6. VERIFY TEST DATA
# ============================================================

print("\nTest set verification:")

print(
    "Saved test samples:",
    len(X_test)
)

print(
    "Recreated test samples:",
    len(X_test_original)
)


# ============================================================
# 7. GET RAW PROBABILITIES
# ============================================================

raw_probabilities = best_model.predict_proba(
    X_test_original
)[:, 1]


# ============================================================
# 8. CREATE CALIBRATED MODEL
# ============================================================

print("\nCreating calibrated model...")


# The saved pipeline already contains preprocessing.
#
# We use a cloned copy so that the original saved model
# remains unchanged.

calibrated_model = CalibratedClassifierCV(
    estimator=clone(best_model),
    method="sigmoid",
    cv=5
)


# ============================================================
# 9. TRAIN CALIBRATED MODEL
# ============================================================

calibrated_model.fit(
    X_train,
    y_train
)


# ============================================================
# 10. CALIBRATED PROBABILITIES
# ============================================================

calibrated_probabilities = (
    calibrated_model.predict_proba(
        X_test_original
    )[:, 1]
)


# ============================================================
# 11. BRIER SCORES
# ============================================================

raw_brier = brier_score_loss(
    y_test_original,
    raw_probabilities
)

calibrated_brier = brier_score_loss(
    y_test_original,
    calibrated_probabilities
)


# ============================================================
# 12. LOG LOSS
# ============================================================

raw_log_loss = log_loss(
    y_test_original,
    raw_probabilities
)

calibrated_log_loss = log_loss(
    y_test_original,
    calibrated_probabilities
)


# ============================================================
# 13. ROC-AUC
# ============================================================

raw_auc = roc_auc_score(
    y_test_original,
    raw_probabilities
)

calibrated_auc = roc_auc_score(
    y_test_original,
    calibrated_probabilities
)
# ============================================================
# 13A. EXPECTED CALIBRATION ERROR (ECE)
# ============================================================

def expected_calibration_error(
    y_true,
    probabilities,
    n_bins=10
):
    """
    Calculate Expected Calibration Error (ECE).

    ECE measures the difference between:
    - average predicted probability
    - observed fraction of positive samples

    Lower ECE indicates better calibration.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    ece = 0.0

    total_samples = len(y_true)

    for i in range(n_bins):

        if i == n_bins - 1:

            mask = (
                (probabilities >= bin_edges[i]) &
                (probabilities <= bin_edges[i + 1])
            )

        else:

            mask = (
                (probabilities >= bin_edges[i]) &
                (probabilities < bin_edges[i + 1])
            )

        if np.sum(mask) == 0:
            continue

        bin_probabilities = probabilities[mask]
        bin_actuals = y_true[mask]

        mean_probability = np.mean(
            bin_probabilities
        )

        observed_frequency = np.mean(
            bin_actuals
        )

        bin_weight = (
            len(bin_actuals) /
            total_samples
        )

        ece += (
            bin_weight *
            abs(
                mean_probability -
                observed_frequency
            )
        )

    return ece


raw_ece = expected_calibration_error(
    y_test_original,
    raw_probabilities
)

calibrated_ece = expected_calibration_error(
    y_test_original,
    calibrated_probabilities
)


# ============================================================
# 14. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATION RESULTS")
print("=" * 70)

print(
    f"{'Metric':<20}"
    f"{'Raw':<15}"
    f"{'Calibrated':<15}"
)

print("-" * 50)

print(
    f"{'Brier Score':<20}"
    f"{raw_brier:<15.6f}"
    f"{calibrated_brier:<15.6f}"
)

print(
    f"{'Log Loss':<20}"
    f"{raw_log_loss:<15.6f}"
    f"{calibrated_log_loss:<15.6f}"
)

print(
    f"{'ROC-AUC':<20}"
    f"{raw_auc:<15.6f}"
    f"{calibrated_auc:<15.6f}"
)
print(
    f"{'ECE':<20}"
    f"{raw_ece:<15.6f}"
    f"{calibrated_ece:<15.6f}"
)

# ============================================================
# 15. CALIBRATION CURVES
# ============================================================

fraction_of_positives_raw, mean_predicted_value_raw = (
    calibration_curve(
        y_test_original,
        raw_probabilities,
        n_bins=10,
        strategy="uniform"
    )
)


fraction_of_positives_calibrated, mean_predicted_value_calibrated = (
    calibration_curve(
        y_test_original,
        calibrated_probabilities,
        n_bins=10,
        strategy="uniform"
    )
)


# ============================================================
# 16. PLOT RELIABILITY DIAGRAM
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.plot(
    mean_predicted_value_raw,
    fraction_of_positives_raw,
    marker="o",
    label="Raw Model"
)

plt.plot(
    mean_predicted_value_calibrated,
    fraction_of_positives_calibrated,
    marker="s",
    label="Calibrated Model"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.xlabel(
    "Mean Predicted Probability"
)

plt.ylabel(
    "Observed Fraction of Positives"
)

plt.title(
    "Probability Reliability Diagram"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "calibration_reliability_curve.png",
    dpi=300
)

plt.show()


# ============================================================
# 17. SAVE RESULTS
# ============================================================

calibration_results = pd.DataFrame({

    "Model": [
        best_model_name,
        best_model_name + " + Calibration"
    ],

    "Brier Score": [
        raw_brier,
        calibrated_brier
    ],

    "Log Loss": [
        raw_log_loss,
        calibrated_log_loss
    ],

    "ROC-AUC": [
        raw_auc,
        calibrated_auc
    ],

    "ECE": [
        raw_ece,
        calibrated_ece
    ]
})


calibration_results.to_csv(
    "calibration_results.csv",
    index=False
)


# ============================================================
# 18. SAVE CALIBRATED MODEL
# ============================================================

calibration_package = {

    "model": calibrated_model,

    "model_name":
        best_model_name,

    "calibration_method":
        "sigmoid",

    "brier_score":
        calibrated_brier,

    "log_loss":
        calibrated_log_loss,

    "roc_auc":
        calibrated_auc
}


joblib.dump(
    calibration_package,
    "calibrated_model.pkl"
)


# ============================================================
# 19. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATION ANALYSIS COMPLETED")
print("=" * 70)

print(
    "Results saved as:",
    "calibration_results.csv"
)

print(
    "Reliability curve saved as:",
    "calibration_reliability_curve.png"
)

print(
    "Calibrated model saved as:",
    "calibrated_model.pkl"
)

print("=" * 70)