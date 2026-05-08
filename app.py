import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, session, url_for

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super-secret-key")

model_path = Path("model.joblib")
amount_scaler_path = Path("amount_scaler.joblib")
time_scaler_path = Path("time_scaler.joblib")
dataset_path = Path("balanced_creditcard.csv")

if not model_path.exists() or not amount_scaler_path.exists() or not time_scaler_path.exists():
    raise FileNotFoundError("Required model or scaler files are missing. Run train_model.py first.")

model = joblib.load(model_path)
amount_scaler = joblib.load(amount_scaler_path)
time_scaler = joblib.load(time_scaler_path)


def login_required(view_func):
    from functools import wraps

    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped_view


def load_dataset_stats():
    if not dataset_path.exists():
        return {}

    df = pd.read_csv(dataset_path)
    fraud_count = int(df["Class"].sum())
    legit_count = len(df) - fraud_count

    return {
        "total_transactions": len(df),
        "fraud_transactions": fraud_count,
        "legit_transactions": legit_count,
        "fraud_ratio": round(fraud_count / len(df) * 100, 2),
        "avg_amount_fraud": round(df.loc[df["Class"] == 1, "Amount"].mean() or 0, 2),
        "avg_amount_legit": round(df.loc[df["Class"] == 0, "Amount"].mean() or 0, 2),
        "avg_time_fraud": round(df.loc[df["Class"] == 1, "Time"].mean() or 0, 2),
        "avg_time_legit": round(df.loc[df["Class"] == 0, "Time"].mean() or 0, 2),
    }


dataset_stats = load_dataset_stats()


@app.route("/")
def home():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        if not email or not password:
            error = "Please enter both email and password."
        else:
            session["user"] = email
            session.setdefault("history", [])
            flash("Welcome back! Your fraud dashboard is ready.", "success")
            return redirect(url_for("dashboard"))

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("user", None)
    session.pop("history", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


def build_feature_vector(amount, time_seconds):
    scaled_amount = amount_scaler.transform(np.array([[amount]]))[0][0]
    scaled_time = time_scaler.transform(np.array([[time_seconds]]))[0][0]
    feature_vector = [0.0] * 28 + [scaled_amount, scaled_time]
    return np.array(feature_vector).reshape(1, -1), scaled_amount, scaled_time


def update_history(row):
    history = session.get("history", [])
    history.insert(0, row)
    session["history"] = history[:5]


@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    prediction = None
    probability = None
    risk_level = None
    summary = None

    if request.method == "POST":
        try:
            amount = float(request.form.get("amount", 0))
            time_value = float(request.form.get("time", 0))

            features, scaled_amount, scaled_time = build_feature_vector(amount, time_value)
            proba = model.predict_proba(features)[0][1]
            probability = round(proba * 100, 1)
            prediction = "Fraud" if proba >= 0.50 else "Legit"
            risk_level = (
                "High risk" if proba >= 0.75 else "Medium risk" if proba >= 0.40 else "Low risk"
            )

            summary = {
                "amount": amount,
                "time": int(time_value),
                "scaled_amount": round(scaled_amount, 4),
                "scaled_time": round(scaled_time, 4),
                "model_type": type(model).__name__,
            }

            update_history(
                {
                    "amount": amount,
                    "time": int(time_value),
                    "prediction": prediction,
                    "probability": f"{probability}%",
                    "risk": risk_level,
                }
            )
        except ValueError:
            prediction = "Invalid input. Please enter valid numeric values."
        except Exception as exc:
            prediction = f"Error: {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        risk_level=risk_level,
        summary=summary,
        stats=dataset_stats,
        history=session.get("history", []),
    )


@app.route("/settings")
@login_required
def settings():
    model_info = {
        "model_name": type(model).__name__,
        "feature_count": int(getattr(model, "n_features_in_", 0)),
        "dataset_available": bool(dataset_stats),
    }
    return render_template("settings.html", model_info=model_info, stats=dataset_stats)


if __name__ == "__main__":
    app.run(debug=True)
