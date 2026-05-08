import joblib

best_model = models['XGBoost']  # Change if RandomForest or LR was better
joblib.dump(best_model, 'fraud_model.pkl')
print("✅ Model saved as fraud_model.pkl")
