# Credit Card Fraud Detection

A simple Flask-based credit card fraud detection app using a trained machine learning model.

## Project Overview

This repository contains:
- `app.py` — Flask web application to input transaction details and display a fraud prediction.
- `train_model.py` — Script to preprocess the dataset, train models, compare performance, and save the best model.
- `balanced_creditcard.csv` — Dataset used for training.
- `model.joblib`, `amount_scaler.joblib`, `time_scaler.joblib` — Saved model and scaler artifacts.
- `templates/` — HTML templates for the web interface.
- `static/` — Static resources including the model comparison plot.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Flask app:

```bash
python app.py
```

3. Open your browser at `http://127.0.0.1:5000/dashboard`.

## Usage

- Enter the transaction `Amount` and `Time` values.
- Submit the form to get a fraud prediction.
- The app returns `Fraud` or `Legit` based on the saved machine learning model.

## Training

To retrain the model from scratch:

```bash
python train_model.py
```

This script:
- loads `balanced_creditcard.csv`
- scales `Amount` and `Time`
- uses SMOTE to balance training data
- trains Logistic Regression, Random Forest, and XGBoost models
- saves the best model as `model.joblib`
- saves scalers for runtime input preprocessing

## Requirements

- Flask
- numpy
- pandas
- scikit-learn
- imbalanced-learn
- xgboost
- joblib
- matplotlib

## Notes

- Make sure `model.joblib`, `amount_scaler.joblib`, and `time_scaler.joblib` are present in the project root before running `app.py`.
- The dataset includes a balanced version of the credit card fraud data.
