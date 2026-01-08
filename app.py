from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("knn_heart_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    if request.method == "POST":
        data = {
            "Age": int(request.form["Age"]),
            "Sex": request.form["Sex"],
            "ChestPainType": request.form["ChestPainType"],
            "RestingBP": int(request.form["RestingBP"]),
            "Cholesterol": int(request.form["Cholesterol"]),
            "FastingBS": int(request.form["FastingBS"]),
            "RestingECG": request.form["RestingECG"],
            "MaxHR": int(request.form["MaxHR"]),
            "ExerciseAngina": request.form["ExerciseAngina"],
            "Oldpeak": float(request.form["Oldpeak"]),
            "ST_Slope": request.form["ST_Slope"]
        }

        df = pd.DataFrame([data])

        proba = model.predict_proba(df)[0][1]

        prediction = "❤️ Heart Disease Detected" if proba >= 0.45 else "✅ No Heart Disease"
        probability = round(proba * 100, 2)

    return render_template("index.html", prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
