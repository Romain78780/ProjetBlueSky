#!/usr/bin/env python3
# src/main.py

import os
import re
import csv
import json
import spacy
import torch
from langdetect import detect, DetectorFactory
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from api.create_session import create_session
from api.search         import search_posts, extract_search_tweets

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
HANDLE_OR_EMAIL = None      # ou ton handle/.env
PASSWORD        = None      # ou ton mot de passe/.env

QUERY        = "politics"
MAX_POSTS    = 2000
PAGE_LIMIT   = 100

# Seuil de décision sur la probabilité P(fake)
# – issu du script find_optimal_threshold.py
THRESHOLD    = 0.65

PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_CSV_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "tweets_all.csv")
CLEAN_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "tweets_clean.csv")

DetectorFactory.seed = 0  # pour reproductibilité

# -------------------------------------------------------------------
# SpaCy pour lemmatisation
# -------------------------------------------------------------------
def load_spacy_model():
    try:
        return spacy.load("fr_core_news_sm", disable=["parser", "ner"])
    except OSError:
        from spacy.cli import download as spacy_download
        print("Modèle fr_core_news_sm manquant, téléchargement…")
        spacy_download("fr_core_news_sm")
        return spacy.load("fr_core_news_sm", disable=["parser", "ner"])

nlp = load_spacy_model()

def clean_text_spacy(text: str) -> str:
    t = text.lower()
    t = re.sub(r"http\S+|www\.\S+", "", t)
    t = re.sub(r"[@#]\w+", "", t)
    t = re.sub(r"\d+", "", t)
    t = re.sub(r"[^\x00-\x7F]", "", t)
    doc = nlp(t)
    return " ".join(tok.lemma_.lower() for tok in doc if tok.is_alpha and not tok.is_stop)

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "unknown"

# -------------------------------------------------------------------
# Chargement du modèle BERTweet fine-tuné (local)
# -------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = os.path.join(PROJECT_ROOT, "src", "models", "bertweet-fake-news")

TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_DIR, use_fast=True, local_files_only=True
)
MODEL = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, local_files_only=True
).to(DEVICE)
MODEL.eval()

def predict_fake_score(text: str) -> float:
    """
    Renvoie la probabilité P(classe 'fake') pour un texte donné.
    """
    inputs = TOKENIZER(
        text,
        truncation=True,
        padding="max_length",
        max_length=96,
        return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(**inputs).logits
        probs  = torch.softmax(logits, dim=-1).cpu().squeeze().tolist()
    # probs = [P(true), P(fake)]
    return float(probs[1])

# -------------------------------------------------------------------
# Sauvegarde CSV
# -------------------------------------------------------------------
def save_to_csv(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "uri","handle","text","createdAt","lang",
                "pred_label","fake_score","record"
            ],
            delimiter="|",
            quotechar='"',
            quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        for rec in records:
            rec["record"] = json.dumps(rec, ensure_ascii=False)
            writer.writerow(rec)

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    # 1) Authentification
    token = create_session(HANDLE_OR_EMAIL, PASSWORD)
    if not token:
        return

    # 2) Recherche + pagination
    raw_posts = search_posts(token, QUERY, MAX_POSTS, PAGE_LIMIT)
    print(f"🔄 Total récupéré : {len(raw_posts)} posts")

    # 3) Extraction
    raw_tweets = extract_search_tweets(raw_posts)
    print(f"📝 Tweets extraits : {len(raw_tweets)}")

    # 4) Détection langue + sauvegarde brute
    for t in raw_tweets:
        t["lang"] = detect_language(t["text"])
    save_to_csv(raw_tweets, RAW_CSV_PATH)
    print(f"✅ {len(raw_tweets)} tweets bruts → {RAW_CSV_PATH}")

    # 5) Nettoyage, filtrage (fr/en), classification à seuil fixe
    clean_tweets = []
    for t in raw_tweets:
        ct   = clean_text_spacy(t["text"])
        lang = detect_language(ct)
        if lang in ("fr", "en"):
            score = predict_fake_score(ct)
            pred  = 1 if score >= THRESHOLD else 0
            clean_tweets.append({
                **t,
                "text":       ct,
                "lang":       lang,
                "pred_label": pred,
                "fake_score": score
            })
    print(f"🔎 Tweets filtrés (FR/EN) : {len(clean_tweets)}")

    # 6) Sauvegarde finale
    save_to_csv(clean_tweets, CLEAN_CSV_PATH)
    print(f"✅ {len(clean_tweets)} tweets classifiés → {CLEAN_CSV_PATH}")

    # 7) Aperçu
    for t in clean_tweets:
        label = "FAKE" if t["pred_label"] else "TRUE"
        print(f"({t['handle']}) {t['createdAt']} [{t['lang']}] "
              f"{label}({t['fake_score']:.2f}) : {t['text']}\n")

if __name__ == "__main__":
    main()
