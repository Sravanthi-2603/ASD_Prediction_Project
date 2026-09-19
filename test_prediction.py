import pandas as pd
import joblib

# Load trained model
package = joblib.load("autism_model.pkl")

model = package["model"]

print("Model loaded:", package["model_name"])

# --------------------------------------------------
# Test input
# --------------------------------------------------

test_data = {
    "A1_Score": [1],
    "A2_Score": [1],
    "A3_Score": [1],
    "A4_Score": [1],
    "A5_Score": [0],
    "A6_Score": [0],
    "A7_Score": [0],
    "A8_Score": [0],
    "A9_Score": [0],
    "A10_Score": [0],

    "age": [25],

    "gender": ["m"],
    "ethnicity": ["White-European"],
    "jundice": ["no"],
    "austim": ["no"],
    "contry_of_res": ["India"],
    "used_app_before": ["no"],
    "relation": ["Self"]
}

input_df = pd.DataFrame(test_data)

# Prediction
prediction = model.predict(input_df)[0]

# Probability
probability = model.predict_proba(input_df)[0]

print("\n" + "=" * 50)
print("TEST PREDICTION")
print("=" * 50)

print("Screening Score:",
      sum(test_data[f"A{i}_Score"][0] for i in range(1, 11)),
      "/ 10")

print("Prediction:",
      "ASD" if prediction == 1 else "NO ASD")

print("NO ASD Probability:", round(probability[0] * 100, 2), "%")
print("ASD Probability:", round(probability[1] * 100, 2), "%")