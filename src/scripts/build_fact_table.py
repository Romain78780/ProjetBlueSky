import os
import re
import csv
import pandas as pd
import nltk
from nltk.corpus import stopwords
from datasets import load_dataset
from transformers import pipeline

# -------------------------------------------------------------------
# Préparation NLTK (stopwords FR & EN)
# -------------------------------------------------------------------
try:
    _ = stopwords.words("english")
    _ = stopwords.words("french")
except LookupError:
    nltk.download("stopwords")
STOPWORDS_FR = set(stopwords.words("french"))
STOPWORDS_EN = set(stopwords.words("english"))

# -------------------------------------------------------------------
# Pipeline pour la détection des émotions (English only)
# -------------------------------------------------------------------
# Remarque : ce modèle renvoie une émotion parmi :
# 'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=False
)

def detect_emotion(text: str) -> tuple[str, float]:
    """
    Détecte l'émotion dominante et sa confiance.
    Tronque le texte à 512 caractères pour le modèle.
    """
    snippet = text[:512]
    res = emotion_pipeline(snippet)[0]
    return res["label"], res["score"]

# -------------------------------------------------------------------
# Mapping des labels LIAR → binaire
# -------------------------------------------------------------------
def map_liar_label(label_str: str) -> int:
    return 0 if label_str in {"true", "mostly-true"} else 1

# -------------------------------------------------------------------
# Extraction de features textuelles simples
# -------------------------------------------------------------------
def extract_features(text: str) -> dict:
    clean = re.sub(r"http\S+|@\w+|#\w+", "", text)
    tokens = clean.split()
    return {
        "n_tokens":    len(tokens),
        "n_stopwords": sum(1 for t in tokens if t.lower() in STOPWORDS_FR or t.lower() in STOPWORDS_EN),
        "n_chars":     len(clean),
        "n_upper":     sum(1 for c in text if c.isupper()),
    }

# -------------------------------------------------------------------
# Construction de la table de faits à partir de LIAR
# -------------------------------------------------------------------
def build_fact_table(output_path="data/processed/fact_table.csv"):
    print("🔄 Chargement du dataset LIAR (trust_remote_code=True)…")
    ds = load_dataset("liar", trust_remote_code=True)

    # 1) Concaténation splits
    dfs = []
    for split in ("train","validation","test"):
        tmp = pd.DataFrame(ds[split])[["statement","label"]].copy()
        tmp["set"] = split
        dfs.append(tmp)
    df = pd.concat(dfs, ignore_index=True)

    # 2) Labels
    label_names = ds["train"].features["label"].names
    df["label_str"] = df["label"].apply(lambda i: label_names[i])
    df["label_bin"] = df["label_str"].apply(map_liar_label)
    df = df.rename(columns={"statement":"text"})

    # 3) Features textuelles
    feats = df["text"].apply(extract_features).apply(pd.Series)
    df = pd.concat([df, feats], axis=1)

    # 4) Sentiment (étoiles) – optionnel, si tu veux garder aussi
    # ... ici on omet ou tu peux réutiliser ton code existant ...

    # 5) Émotions
    print("🔄 Détection des émotions (peut prendre du temps)…")
    emotions = df["text"].apply(detect_emotion).apply(pd.Series)
    emotions.columns = ["emotion_label","emotion_score"]
    df = pd.concat([df, emotions], axis=1)

    # 6) Sauvegarde CSV avec '|' et QUOTE_ALL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cols = [
        "text",
        "label_str",
        "label_bin",
        "set",
        "n_tokens",
        "n_stopwords",
        "n_chars",
        "n_upper",
        "emotion_label",
        "emotion_score"
    ]
    df[cols].to_csv(
        output_path,
        sep="|",
        index=False,
        encoding="utf-8",
        quotechar='"',
        quoting=csv.QUOTE_ALL
    )
    print(f"✅ Table de faits avec émotions : {df.shape[0]} lignes → {output_path}")

if __name__ == "__main__":
    build_fact_table()
