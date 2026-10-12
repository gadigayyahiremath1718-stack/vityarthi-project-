"""Breast Cancer Diagnosis Classifier
Student: Gadigayya Hiremath (24BEY10151)
Usage:
    python cancer_classifier.py train
    python cancer_classifier.py predict --sample 0     # predict for row 0 of the dataset
"""
import argparse
import json
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

MODEL_PATH = "best_model.joblib"
SEED = 42


def load_data():
    data = load_breast_cancer()
    return data.data, data.target, list(data.target_names), list(data.feature_names)


def train():
    X, y, names, _ = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    models = {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    }
    best_name, best_model, best_f1 = None, None, -1
    for name, model in models.items():
        cv = cross_val_score(model, X_tr, y_tr, cv=5, scoring="f1").mean()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        f1 = f1_score(y_te, pred)
        print(f"\n== {name} ==")
        print(f"CV F1 (5-fold): {cv:.4f}")
        print(f"Accuracy : {accuracy_score(y_te, pred):.4f}")
        print(f"Precision: {precision_score(y_te, pred):.4f}")
        print(f"Recall   : {recall_score(y_te, pred):.4f}")
        print(f"F1       : {f1:.4f}")
        print("Confusion matrix [rows=true, cols=pred]:\n", confusion_matrix(y_te, pred))
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, model, f1
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nBest model: {best_name} (F1={best_f1:.4f}) saved to {MODEL_PATH}")


def predict(sample_idx):
    X, y, names, _ = load_data()
    if not 0 <= sample_idx < len(X):
        raise SystemExit(f"Error: --sample must be between 0 and {len(X)-1}")
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise SystemExit("Error: model not found. Run 'python cancer_classifier.py train' first.")
    proba = model.predict_proba(X[sample_idx].reshape(1, -1))[0]
    result = {
        "sample": sample_idx,
        "predicted": names[int(np.argmax(proba))],
        "confidence": round(float(proba.max()), 4),
        "actual": names[int(y[sample_idx])],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Breast cancer diagnosis classifier")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    sp = sub.add_parser("predict")
    sp.add_argument("--sample", type=int, required=True)
    a = p.parse_args()
    train() if a.cmd == "train" else predict(a.sample)
