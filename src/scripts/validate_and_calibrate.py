#!/usr/bin/env python3
# src/scripts/validate_and_calibrate.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
import joblib

# -------------------------------------------------------------------
# Chemins
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FACT_PATH    = os.path.join(PROJECT_ROOT, "src", "data", "processed", "fact_table.csv")
MODEL_OUT    = os.path.join(PROJECT_ROOT, "src", "models", "fake_news_clf_calibrated.pkl")

# -------------------------------------------------------------------
# Chargement des données
# -------------------------------------------------------------------
def load_data():
    df = pd.read_csv(FACT_PATH, sep="|", quotechar='"', engine="python")
    # On s'attend à une colonne binaire "label_bin" (1=fake, 0=vrai)
    if "label_bin" in df.columns:
        y = df["label_bin"].astype(int)
    else:
        raise KeyError("Votre fact_table.csv doit contenir une colonne 'label_bin'")
    X = df["text"].fillna("")
    return X, y

# -------------------------------------------------------------------
# Construction du pipeline
# -------------------------------------------------------------------
def build_pipeline():
    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=5
    )
    base_clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced"
    )
    calibrated = CalibratedClassifierCV(
        estimator=base_clf,
        method="sigmoid",
        cv=5
    )
    pipe = Pipeline([
        ("tfidf", tfidf),
        ("clf", calibrated)
    ])
    return pipe

# -------------------------------------------------------------------
# Routine principale
# -------------------------------------------------------------------
def main():
    # 1) Charger les données
    X, y = load_data()
    print(f"Chargé {len(y)} exemples → {y.sum()} fake / {len(y)-y.sum()} vrais")

    # 2) Cross-validation 5-fold
    pipe = build_pipeline()
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("⏳ Lancement de la 5-fold cross-validation…")
    scores = cross_validate(
        pipe, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    # 3) Affichage des résultats CV
    print(f"accuracy  : {scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f}")
    print(f"precision : {scores['test_precision'].mean():.3f} ± {scores['test_precision'].std():.3f}")
    print(f"recall    : {scores['test_recall'].mean():.3f} ± {scores['test_recall'].std():.3f}")
    print(f"f1        : {scores['test_f1'].mean():.3f} ± {scores['test_f1'].std():.3f}")
    print(f"roc_auc   : {scores['test_roc_auc'].mean():.3f} ± {scores['test_roc_auc'].std():.3f}")

    # 4) Séparation train / test stratifiée (20 %)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # 5) Entraînement final sur le train set
    pipe.fit(X_train, y_train)

    # 6) Prédictions et scores sur le test set
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    print("\n=== Évaluation sur le jeu de test (20 %) ===")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1       :", f1_score(y_test, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_test, y_prob))

    # Optionnel : afficher le classification_report
    print("\n" + classification_report(y_test, y_pred, target_names=["Vrai","Fake"]))

    # 7) Sauvegarde du modèle calibré
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(pipe, MODEL_OUT)
    print(f"\n✅ Modèle calibré enregistré dans {MODEL_OUT}")

if __name__ == "__main__":
    main()
