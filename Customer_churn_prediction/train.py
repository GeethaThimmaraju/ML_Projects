# src/train.py
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

RAW_DATA_PATH = "../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = "../data/processed/churn_processed.csv"
MODEL_PATH = "../models/churn_model.pkl"

os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def preprocess(df):
    df = df.copy()
    # TotalCharges -> numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    # drop id
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    # map binaries
    map_yesno = {"Yes": 1, "No": 0}
    for c in ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]:
        if c in df.columns:
            df[c] = df[c].map(map_yesno)
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    # one-hot encode categorical features used in original dataset
    categorical = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]
    cols = [c for c in categorical if c in df.columns]
    if cols:
        df = pd.get_dummies(df, columns=cols, drop_first=True)
    return df

def main():
    print("Loading raw data:", RAW_DATA_PATH)
    df = pd.read_csv(RAW_DATA_PATH)
    print("Raw shape:", df.shape)

    print("Preprocessing...")
    df_proc = preprocess(df)
    df_proc.to_csv(PROCESSED_DATA_PATH, index=False)
    print("Saved processed CSV to:", PROCESSED_DATA_PATH)

    # Prepare X,y
    if "Churn" not in df_proc.columns:
        raise ValueError("Target column 'Churn' not found after preprocessing.")
    X = df_proc.drop(columns=["Churn"])
    y = df_proc["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Training model (XGBoost)...")
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    # save model and also save training columns for alignment
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    # save training columns
    cols_path = os.path.join(os.path.dirname(MODEL_PATH), "training_columns.txt")
    with open(cols_path, "w") as f:
        f.write(",".join(X.columns.tolist()))
    print("Saved model to:", MODEL_PATH)
    print("Saved training columns to:", cols_path)

if __name__ == "__main__":
    main()
