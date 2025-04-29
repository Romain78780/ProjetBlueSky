#!/usr/bin/env python3
# src/main.py

import os
import re
import csv
import json
import requests
import spacy
from langdetect import detect, DetectorFactory
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from api.create_session import create_session

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
ACTOR          = "andyrichter.co"            # handle dont tu veux l’historique
MAX_TWEETS     = 250
PAGE_LIMIT     = 100

PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_CSV_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "tweets_all.csv")
CLEAN_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "tweets_clean.csv")

DetectorFactory.seed = 0  # pour la reproductibilité de la détection de langue

# -------------------------------------------------------------------
# SpaCy pour lemmatisation
# -------------------------------------------------------------------
def load_spacy_model():
    try:
        return spacy.load("fr_core_news_sm", disable=["parser","ner"])
    except OSError:
        from spacy.cli import download as spacy_download
        print("Modèle fr_core_news_sm manquant, téléchargement en cours…")
        spacy_download("fr_core_news_sm")
        return spacy.load("fr_core_news_sm", disable=["parser","ner"])

nlp = load_spacy_model()

def clean_text_spacy(text: str) -> str:
    t = text.lower()
    t = re.sub(r"http\S+|www\.\S+", "", t)
    t = re.sub(r"[@#]\w+", "", t)
    t = re.sub(r"\d+", "", t)
    t = re.sub(r"[^\x00-\x7F]", "", t)
    doc = nlp(t)
    return " ".join(tok.lemma_.lower() for tok in doc if tok.is_alpha and not tok.is_stop)

# -------------------------------------------------------------------
# Langue
# -------------------------------------------------------------------
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "unknown"

# -------------------------------------------------------------------
# Pagination sur getAuthorFeed
# -------------------------------------------------------------------
def get_author_feed(token: str, actor: str, max_tweets: int) -> list[dict]:
    all_posts = []
    cursor = None
    headers = {"Authorization": f"Bearer {token}"}
    while len(all_posts) < max_tweets:
        params = {"actor": actor, "limit": PAGE_LIMIT}
        if cursor:
            params["cursor"] = cursor
        url = "https://bsky.social/xrpc/app.bsky.feed.getAuthorFeed"
        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            print(f"❌ getAuthorFeed failed [{resp.status_code}]: {resp.text}")
            break
        data = resp.json()
        feed = data.get("feed", [])
        print(f"Page récupérée: {len(feed)} posts (cursor={data.get('cursor')})")
        if not feed:
            break
        all_posts.extend(feed)
        cursor = data.get("cursor")
        if not cursor:
            break
    return all_posts[:max_tweets]

# -------------------------------------------------------------------
# Extraction
# -------------------------------------------------------------------
def extract_tweets(items: list[dict]) -> list[dict]:
    tweets = []
    for entry in items:
        post = entry.get("post", {})
        rec  = post.get("record", {})
        auth = post.get("author", {})
        tweets.append({
            "uri":       post.get("uri", ""),
            "handle":    auth.get("handle", ""),
            "text":      rec.get("text", ""),
            "createdAt": rec.get("createdAt", "")
        })
    return tweets

# -------------------------------------------------------------------
# Classification BERTweet
# -------------------------------------------------------------------
# on charge le modèle fine-tuné
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained(
    os.path.join(PROJECT_ROOT, "models", "bertweet-fake-news"),
    use_fast=True
)
MODEL     = AutoModelForSequenceClassification.from_pretrained(
    os.path.join(PROJECT_ROOT, "models", "bertweet-fake-news")
).to(DEVICE)
MODEL.eval()

def classify_text(text: str) -> tuple[int, float]:
    """
    renvoie (pred_label, fake_score)
    pred_label: 1 = fake, 0 = vrai
    fake_score: probabilité de la classe 'fake'
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
    # probs = [p_true, p_fake]
    return (int(probs[1] > probs[0]), float(probs[1]))

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
    token = create_session()
    if not token:
        return

    # 2) Récupération paginée
    raw_items = get_author_feed(token, ACTOR, MAX_TWEETS)
    print(f"🔄 Total raw items: {len(raw_items)}")

    # 3) Extraction des champs bruts
    raw_tweets = extract_tweets(raw_items)
    print(f"📝 Tweets extraits : {len(raw_tweets)}")

    # 4) Détection langue + sauvegarde brute
    for t in raw_tweets:
        t["lang"] = detect_language(t["text"])
    save_to_csv(raw_tweets, RAW_CSV_PATH)
    print(f"✅ {len(raw_tweets)} tweets bruts sauvegardés → {RAW_CSV_PATH}")

    # 5) Nettoyage + filtrage (fr/en)
    clean_tweets = []
    for t in raw_tweets:
        clean = clean_text_spacy(t["text"])
        lang  = detect_language(clean)
        if lang in ("fr","en"):
            rec = {
                **t,
                "text": clean,
                "lang": lang
            }
            # 6) Classification
            pred, score = classify_text(clean)
            rec["pred_label"] = pred
            rec["fake_score"] = score
            clean_tweets.append(rec)

    print(f"🔎 Tweets filtrés (fr/en): {len(clean_tweets)}")

    # 7) Sauvegarde nettoyée + prédictions
    save_to_csv(clean_tweets, CLEAN_CSV_PATH)
    print(f"✅ {len(clean_tweets)} tweets validés et classifiés → {CLEAN_CSV_PATH}")

    # 8) Aperçu
    for t in clean_tweets:
        label = "FAKE" if t["pred_label"] == 1 else "TRUE"
        print(f"({t['handle']}) {t['createdAt']} [{t['lang']}] {label}({t['fake_score']:.2f}) : {t['text']}\n")

if __name__ == "__main__":
    main()
