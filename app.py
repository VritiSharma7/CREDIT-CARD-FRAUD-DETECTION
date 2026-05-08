from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load the model and scalers
model = joblib.load("model.joblib")
amount_scaler = joblib.load("amount_scaler.joblib")
time_scaler = joblib.load("time_scaler.joblib")

@app.route("/")
def home():
    return redirect(url_for("dashboard"))

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    prediction = None
    if request.method == "POST":
        try:
            # Get form data safely
            amount = float(request.form.get("amount", 0))
            time_val = float(request.form.get("time", 0))

            # Scale inputs
            scaled_amount = amount_scaler.transform([[amount]])[0][0]
            scaled_time = time_scaler.transform([[time_val]])[0][0]

            # Predict
            result = model.predict([[scaled_time, scaled_amount]])[0]
            prediction = "Fraud" if result == 1 else "Legit"

        except Exception as e:
            prediction = f"ERROR: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
