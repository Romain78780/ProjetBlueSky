#!/usr/bin/env python3
# src/scripts/find_optimal_threshold.py

import os
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import roc_curve, precision_recall_curve

# 1) Localisation des fichiers
PROJECT_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FACT_PATH         = os.path.join(PROJECT_ROOT, "src", "data", "fact_table.csv")
CALIBRATED_MODEL  = os.path.join(PROJECT_ROOT, "src", "models", "fake_news_clf_calibrated.pkl")

# 2) Lecture de la table de faits
print("🔄 Chargement de la fact_table…")
df = pd.read_csv(FACT_PATH, sep="|", quotechar='"')
y_true = df["label_bin"].values

# 3) Chargement du modèle calibré (pipeline TF–IDF + Logistic + calibration)
print("🔄 Chargement du modèle calibré…")
clf = load(CALIBRATED_MODEL)

# 4) Calcul des probabilités P(fake)
print("🔄 Prédiction des probabilités…")
# on passe directement la colonne 'text' au pipeline
probs = clf.predict_proba(df["text"].tolist())[:, 1]

# 5) Seuil optimal selon Youden (maximiser tpr–fpr)
fpr, tpr, thresh_roc = roc_curve(y_true, probs)
youden = tpr - fpr
opt_idx_roc       = np.argmax(youden)
opt_threshold_roc = thresh_roc[opt_idx_roc]

# 6) Seuil optimisant la F1 (precision-recall)
precision, recall, thresh_pr = precision_recall_curve(y_true, probs)
f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
opt_idx_pr       = np.nanargmax(f1_scores)
opt_threshold_pr = thresh_pr[opt_idx_pr]

# 7) Affichage
print(f"✅ Seuil optimal (Youden’s J)   : {opt_threshold_roc:.3f}")
print(f"✅ Seuil optimal (max F1)       : {opt_threshold_pr:.3f}")
