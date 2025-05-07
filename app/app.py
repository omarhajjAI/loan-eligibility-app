from flask import Flask, request, render_template
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

# Load pipeline
MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'pipeline.joblib'
pipeline = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Collect form input
            input_data = {
                "Gender": request.form["Gender"],
                "Married": request.form["Married"],
                "Dependents": request.form["Dependents"],
                "Education": request.form["Education"],
                "Self_Employed": request.form["Self_Employed"],
                "ApplicantIncome": float(request.form["ApplicantIncome"]),
                "CoapplicantIncome": float(request.form["CoapplicantIncome"]),
                "LoanAmount": float(request.form["LoanAmount"]),
                "Loan_Amount_Term": float(request.form["Loan_Amount_Term"]),
                "Credit_History": float(request.form["Credit_History"]),
                "Property_Area": request.form["Property_Area"]
            }

            df = pd.DataFrame([input_data])
            prediction = pipeline.predict(df)[0]
            result = "Eligible for Loan" if prediction == 1 else "Not Eligible"

            return render_template("index.html", prediction_text=result)

        except Exception as e:
            return render_template("index.html", prediction_text=f"Error: {str(e)}")

    return render_template("index.html", prediction_text=None)

if __name__ == '__main__':
    app.run(debug=True)
