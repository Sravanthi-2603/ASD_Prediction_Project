from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

package = joblib.load("autism_model.pkl")

model = package["model"]

print("=" * 60)
print("ASD PREDICTION SYSTEM")
print("=" * 60)
print("Loaded model:", package["model_name"])


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # --------------------------------------------------------
    # A1-A10 screening questions
    # --------------------------------------------------------

    scores = []

    for i in range(1, 11):

        answer = request.form.get(f"A{i}")

        if answer == "Yes":
            score = 1
        else:
            score = 0

        scores.append(score)

    # --------------------------------------------------------
    # Calculate screening score
    # --------------------------------------------------------

    total_score = sum(scores)

    # --------------------------------------------------------
    # Personal information
    # --------------------------------------------------------

    age = request.form.get("age")

    gender = request.form.get("gender")
    ethnicity = request.form.get("ethnicity")
    jundice = request.form.get("jundice")
    austim = request.form.get("austim")
    contry_of_res = request.form.get("contry_of_res")
    used_app_before = request.form.get("used_app_before")
    relation = request.form.get("relation")

    # Convert age to numeric
    try:
        age = float(age)
    except (TypeError, ValueError):
        age = None

    # --------------------------------------------------------
    # Create input DataFrame
    # --------------------------------------------------------

    input_data = {
        "A1_Score": [scores[0]],
        "A2_Score": [scores[1]],
        "A3_Score": [scores[2]],
        "A4_Score": [scores[3]],
        "A5_Score": [scores[4]],
        "A6_Score": [scores[5]],
        "A7_Score": [scores[6]],
        "A8_Score": [scores[7]],
        "A9_Score": [scores[8]],
        "A10_Score": [scores[9]],
        "age": [age],
        "gender": [gender],
        "ethnicity": [ethnicity],
        "jundice": [jundice],
        "austim": [austim],
        "contry_of_res": [contry_of_res],
        "used_app_before": [used_app_before],
        "relation": [relation]
    }

    input_df = pd.DataFrame(input_data)

    # --------------------------------------------------------
    # ML Prediction
    # --------------------------------------------------------

    probabilities = model.predict_proba(input_df)[0]

    no_asd_probability = probabilities[0] * 100
    asd_probability = probabilities[1] * 100

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if asd_probability >= 50:
        prediction_text = "ASD"
    else:
        prediction_text = "NO ASD"

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    return render_template(
        "result.html",
        prediction=prediction_text,
        screening_score=total_score,
        asd_probability=round(asd_probability, 2),
        no_asd_probability=round(no_asd_probability, 2)
    )
# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)