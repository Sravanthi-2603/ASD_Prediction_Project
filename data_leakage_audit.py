import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score


# ============================================================
# DATA LEAKAGE / REDUNDANCY AUDIT
# ============================================================

DATA_PATH = "dataset/autism_screening.csv"

print("=" * 80)
print("DATA LEAKAGE / REDUNDANCY AUDIT")
print("=" * 80)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()

print("\nDataset shape:", df.shape)

print("\nColumns:")
for i, col in enumerate(df.columns, start=1):
    print(f"{i:2d}. {col}")


# ============================================================
# 2. BASIC DATA QUALITY
# ============================================================

print("\n" + "=" * 80)
print("1. MISSING VALUES")
print("=" * 80)

missing = df.isnull().sum()

missing_df = pd.DataFrame({
    "Column": missing.index,
    "Missing_Count": missing.values,
    "Missing_Percentage": (
        missing.values / len(df) * 100
    )
})

print(
    missing_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# 3. EXACT DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 80)
print("2. EXACT DUPLICATE ROWS")
print("=" * 80)

duplicate_rows = df.duplicated().sum()

print("Exact duplicate rows:", duplicate_rows)

if duplicate_rows > 0:
    print("\nDuplicate rows:")
    print(
        df[df.duplicated(keep=False)]
        .sort_values(by=df.columns.tolist())
        .head(20)
        .to_string(index=False)
    )
else:
    print("No exact duplicate rows found.")


# ============================================================
# 4. DUPLICATES EXCLUDING TARGET
# ============================================================

print("\n" + "=" * 80)
print("3. DUPLICATE FEATURE PATTERNS")
print("=" * 80)

TARGET = "Class/ASD"

feature_columns = [
    col for col in df.columns
    if col != TARGET
]

duplicate_feature_mask = df.duplicated(
    subset=feature_columns,
    keep=False
)

duplicate_feature_count = duplicate_feature_mask.sum()

print(
    "Rows belonging to duplicate feature patterns:",
    duplicate_feature_count
)

if duplicate_feature_count > 0:

    duplicate_feature_groups = (
        df[duplicate_feature_mask]
        .groupby(feature_columns, dropna=False)
        .size()
        .reset_index(name="Count")
    )

    print(
        "\nNumber of duplicated feature groups:",
        len(duplicate_feature_groups)
    )

    print("\nFirst duplicated feature groups:")

    print(
        duplicate_feature_groups
        .sort_values("Count", ascending=False)
        .head(20)
        .to_string(index=False)
    )

else:

    print("No duplicate feature patterns found.")


# ============================================================
# 5. SAME FEATURES WITH DIFFERENT TARGETS
# ============================================================

print("\n" + "=" * 80)
print("4. CONFLICTING DUPLICATE FEATURE PATTERNS")
print("=" * 80)

if duplicate_feature_count > 0:

    grouped_targets = (
        df.groupby(
            feature_columns,
            dropna=False
        )[TARGET]
        .nunique()
        .reset_index(name="Unique_Targets")
    )

    conflicting_groups = grouped_targets[
        grouped_targets["Unique_Targets"] > 1
    ]

    print(
        "Feature patterns associated with multiple target values:",
        len(conflicting_groups)
    )

    if len(conflicting_groups) > 0:
        print(
            conflicting_groups.head(20)
            .to_string(index=False)
        )
    else:
        print(
            "No feature pattern was found with conflicting target labels."
        )

else:

    print(
        "No duplicated feature patterns available for conflict analysis."
    )


# ============================================================
# 6. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("5. TARGET DISTRIBUTION")
print("=" * 80)

target_clean = (
    df[TARGET]
    .astype(str)
    .str.strip()
    .str.upper()
)

print(
    target_clean.value_counts(dropna=False)
)

print("\nTarget percentages:")

print(
    (
        target_clean.value_counts(normalize=True, dropna=False)
        * 100
    ).round(2)
)


# ============================================================
# 7. CHECK TARGET-LIKE COLUMNS
# ============================================================

print("\n" + "=" * 80)
print("6. POTENTIAL TARGET-LIKE / DERIVED COLUMNS")
print("=" * 80)

target_terms = [
    "class",
    "asd",
    "autism",
    "result",
    "score",
    "target",
    "label",
    "diagnosis",
    "outcome"
]

potential_columns = []

for col in df.columns:

    col_lower = col.lower()

    matched_terms = [
        term for term in target_terms
        if term in col_lower
    ]

    if matched_terms:

        potential_columns.append({
            "Column": col,
            "Matched_Terms": ", ".join(matched_terms)
        })


potential_columns_df = pd.DataFrame(
    potential_columns
)

if len(potential_columns_df) > 0:

    print(
        potential_columns_df.to_string(index=False)
    )

else:

    print("No obviously target-like column names found.")


# ============================================================
# 8. SPECIFIC CHECK OF 'result'
# ============================================================

print("\n" + "=" * 80)
print("7. 'result' COLUMN AUDIT")
print("=" * 80)

if "result" in df.columns:

    print("\nUnique result values:")
    print(
        df["result"]
        .value_counts(dropna=False)
    )

    print("\nCross-tabulation: result vs Class/ASD")

    result_target_table = pd.crosstab(
        df["result"],
        target_clean,
        margins=True
    )

    print(result_target_table)

    print("\nResult values within each target class:")

    print(
        pd.crosstab(
            df["result"],
            target_clean,
            normalize="columns"
        ).round(4)
    )

else:

    print("'result' column not found.")


# ============================================================
# 9. A1-A10 SCORE AUDIT
# ============================================================

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

print("\n" + "=" * 80)
print("8. A1-A10 SCREENING SCORE AUDIT")
print("=" * 80)


available_screening = [
    col for col in SCREENING_FEATURES
    if col in df.columns
]


screening_df = df[
    available_screening
].copy()

screening_df["Target"] = (
    target_clean
    .map({
        "NO": 0,
        "YES": 1
    })
)


print("\nA1-A10 descriptive statistics:")

print(
    screening_df[
        available_screening
    ].describe().round(4)
)


# Total screening score

screening_df["Total_Screening_Score"] = (
    screening_df[
        available_screening
    ].sum(axis=1)
)


print("\nTotal screening score distribution:")

print(
    screening_df[
        "Total_Screening_Score"
    ].value_counts()
    .sort_index()
)


# ============================================================
# 10. TOTAL SCORE VS TARGET
# ============================================================

print("\n" + "=" * 80)
print("9. TOTAL SCREENING SCORE VS TARGET")
print("=" * 80)

score_target_table = pd.crosstab(
    screening_df["Total_Screening_Score"],
    screening_df["Target"]
)

print(score_target_table)


print("\nMean total screening score by target:")

print(
    screening_df
    .groupby("Target")["Total_Screening_Score"]
    .agg(["count", "mean", "std", "min", "max"])
    .round(4)
)


# ============================================================
# 11. CHECK DETERMINISTIC SCORE-BASED TARGET RELATIONSHIP
# ============================================================

print("\n" + "=" * 80)
print("10. DETERMINISTIC TOTAL-SCORE RELATIONSHIP")
print("=" * 80)

score_target_uniques = (
    screening_df
    .groupby("Total_Screening_Score")["Target"]
    .nunique()
)

deterministic_scores = score_target_uniques[
    score_target_uniques == 1
]

mixed_scores = score_target_uniques[
    score_target_uniques > 1
]

print(
    "Score values having only one target class:",
    len(deterministic_scores)
)

print(
    "Score values containing both target classes:",
    len(mixed_scores)
)

if len(mixed_scores) == 0:

    print(
        "\nIMPORTANT: Every observed total screening score maps "
        "to only one target class in this dataset."
    )

    print(
        "This indicates a very strong deterministic relationship "
        "between the questionnaire score and the target."
    )

else:

    print(
        "\nSome total screening scores contain both target classes."
    )


# ============================================================
# 12. INDIVIDUAL A1-A10 TARGET ASSOCIATION
# ============================================================

print("\n" + "=" * 80)
print("11. INDIVIDUAL A1-A10 TARGET ASSOCIATION")
print("=" * 80)

association_results = []

for feature in available_screening:

    temp = pd.crosstab(
        df[feature],
        screening_df["Target"],
        normalize="index"
    )

    if 1 in temp.columns:

        yes_rates = temp[1]

        max_yes_rate = yes_rates.max()
        min_yes_rate = yes_rates.min()

    else:

        max_yes_rate = np.nan
        min_yes_rate = np.nan

    try:

        mutual_information = mutual_info_score(
            screening_df["Target"],
            df[feature]
        )

    except Exception:

        mutual_information = np.nan

    association_results.append({

        "Feature": feature,

        "Min_ASD_Rate":
            min_yes_rate,

        "Max_ASD_Rate":
            max_yes_rate,

        "Mutual_Information":
            mutual_information
    })


association_df = pd.DataFrame(
    association_results
)

association_df = association_df.sort_values(
    "Mutual_Information",
    ascending=False
)

print(
    association_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 13. TRAIN / TEST OVERLAP AUDIT
# ============================================================

print("\n" + "=" * 80)
print("12. TRAIN / TEST FEATURE OVERLAP")
print("=" * 80)

X = df.drop(columns=[TARGET])

y = (
    target_clean
    .map({
        "NO": 0,
        "YES": 1
    })
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))


# Convert every row into a comparable string representation.
# Using tuple() avoids pandas' mixed-type join problem.

train_feature_keys = X_train.apply(
    lambda row: tuple(row.fillna("<MISSING>").astype(str)),
    axis=1
)

test_feature_keys = X_test.apply(
    lambda row: tuple(row.fillna("<MISSING>").astype(str)),
    axis=1
)

train_key_set = set(train_feature_keys)
test_key_set = set(test_feature_keys)

overlap = train_key_set.intersection(test_key_set)


train_key_set = set(
    train_feature_keys
)

test_key_set = set(
    test_feature_keys
)

overlap = train_key_set.intersection(
    test_key_set
)

print(
    "\nExact feature-pattern overlap between train and test:",
    len(overlap)
)

if len(overlap) > 0:

    print(
        "WARNING: Some identical feature patterns occur in both "
        "training and testing data."
    )

else:

    print(
        "No exact complete-feature pattern occurs in both "
        "training and testing sets."
    )


# ============================================================
# 14. SCREENING-ONLY TRAIN / TEST OVERLAP
# ============================================================

print("\n" + "=" * 80)
print("13. A1-A10 PATTERN OVERLAP BETWEEN TRAIN AND TEST")
print("=" * 80)

train_screening_keys = (
    X_train[
        available_screening
    ]
    .fillna("<MISSING>")
    .astype(str)
    .agg("|".join, axis=1)
)

test_screening_keys = (
    X_test[
        available_screening
    ]
    .fillna("<MISSING>")
    .astype(str)
    .agg("|".join, axis=1)
)

train_screening_set = set(
    train_screening_keys
)

test_screening_set = set(
    test_screening_keys
)

screening_overlap = (
    train_screening_set
    .intersection(test_screening_set)
)

print(
    "A1-A10 patterns occurring in both train and test:",
    len(screening_overlap)
)

if len(screening_overlap) > 0:

    print(
        "NOTE: Repeated questionnaire response patterns exist "
        "across train and test."
    )

else:

    print(
        "No identical A1-A10 response pattern occurs in both splits."
    )


# ============================================================
# 15. SCREENING PATTERN TARGET CONSISTENCY
# ============================================================

print("\n" + "=" * 80)
print("14. A1-A10 PATTERN TARGET CONSISTENCY")
print("=" * 80)

screening_pattern_target = (
    df.groupby(
        available_screening
    )[TARGET]
    .nunique()
    .reset_index(name="Unique_Targets")
)

conflicting_screening_patterns = (
    screening_pattern_target[
        screening_pattern_target["Unique_Targets"] > 1
    ]
)

print(
    "Unique A1-A10 patterns:",
    len(screening_pattern_target)
)

print(
    "A1-A10 patterns with conflicting target labels:",
    len(conflicting_screening_patterns)
)

if len(conflicting_screening_patterns) > 0:

    print(
        "\nExamples of conflicting screening patterns:"
    )

    print(
        conflicting_screening_patterns
        .head(20)
        .to_string(index=False)
    )

else:

    print(
        "Every observed A1-A10 response pattern maps to "
        "a single target class."
    )


# ============================================================
# 16. UNIQUE VALUES PER COLUMN
# ============================================================

print("\n" + "=" * 80)
print("15. UNIQUE VALUES PER COLUMN")
print("=" * 80)

unique_summary = pd.DataFrame({
    "Column": df.columns,
    "Unique_Values": [
        df[col].nunique(dropna=False)
        for col in df.columns
    ]
})

print(
    unique_summary.to_string(index=False)
)


# ============================================================
# 17. SAVE AUDIT RESULTS
# ============================================================

missing_df.to_csv(
    "leakage_audit_missing_values.csv",
    index=False
)

association_df.to_csv(
    "leakage_audit_screening_association.csv",
    index=False
)

unique_summary.to_csv(
    "leakage_audit_unique_values.csv",
    index=False
)

if duplicate_feature_count > 0:

    duplicate_feature_groups.to_csv(
        "leakage_audit_duplicate_feature_patterns.csv",
        index=False
    )

if len(conflicting_screening_patterns) > 0:

    conflicting_screening_patterns.to_csv(
        "leakage_audit_conflicting_screening_patterns.csv",
        index=False
    )


# ============================================================
# 18. FINAL AUDIT SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL AUDIT SUMMARY")
print("=" * 80)

print(
    f"Dataset rows: {len(df)}"
)

print(
    f"Dataset columns: {len(df.columns)}"
)

print(
    f"Exact duplicate rows: {duplicate_rows}"
)

print(
    f"Duplicate feature-pattern rows: "
    f"{duplicate_feature_count}"
)

print(
    f"Train/test complete-feature overlap: "
    f"{len(overlap)}"
)

print(
    f"Train/test A1-A10 pattern overlap: "
    f"{len(screening_overlap)}"
)

print(
    f"Conflicting A1-A10 patterns: "
    f"{len(conflicting_screening_patterns)}"
)

print(
    f"Unique A1-A10 patterns: "
    f"{len(screening_pattern_target)}"
)

print(
    f"Total-score values with mixed targets: "
    f"{len(mixed_scores)}"
)

print("\nAudit files saved successfully.")

print("\n" + "=" * 80)
print("DATA LEAKAGE / REDUNDANCY AUDIT COMPLETED")
print("=" * 80)