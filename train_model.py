# --- Step 1: Import libraries ---
import pandas as pd
import warnings
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# --- Step 2: Load data ---
df = pd.read_csv("balanced_creditcard.csv")

# --- Step 3: Preprocess ---
amount_scaler = StandardScaler()
time_scaler = StandardScaler()

df['Scaled_Amount'] = amount_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Scaled_Time'] = time_scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# ✅ Save both scalers as .joblib
joblib.dump(amount_scaler, "amount_scaler.joblib")
joblib.dump(time_scaler, "time_scaler.joblib")
print("✅ Saved amount_scaler.joblib and time_scaler.joblib")

# --- Step 4: Prepare X and y using ONLY Scaled_Time and Scaled_Amount ---
# Drop original Amount and Time, keep scaled + V1 to V28
X = df.drop(['Class', 'Time', 'Amount'], axis=1)
y = df['Class']

# --- Step 5: Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# --- Step 6: Apply SMOTE ---
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# --- Step 7: Train Models ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = {}
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    results[name] = auc
    print(f"{name}: ROC AUC = {auc:.4f}")

# --- Step 8: Save Best Model ---
best_model_name = max(results, key=results.get)
joblib.dump(models[best_model_name], "model.joblib")
print(f"✅ Saved best model: {best_model_name} as model.joblib")

# --- Step 9: Save Model Comparison Chart ---
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color=['blue', 'green', 'orange'])
plt.title("Model Comparison (ROC AUC)")
plt.ylabel("ROC AUC Score")
plt.tight_layout()
plt.savefig("static/model_comparison.png")
print("✅ Model comparison chart saved as static/model_comparison.png")
