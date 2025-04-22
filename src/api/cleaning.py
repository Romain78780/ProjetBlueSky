#!/usr/bin/env python3
# main.py

import os
import csv
import json
import re
import spacy
import requests
from langdetect import detect, DetectorFactory

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# Remplace par tes identifiants Bluesky (ou charge-les depuis .env)
HANDLE_OR_EMAIL = "ton.handle.bsky.social"
PASSWORD        = "TON_MOT_DE_PASSE"

# Nombre de tweets à récupérer par page
TIMELINE_LIMIT = 50

# Chemins de sortie
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "tweets_all.csv")
CLEAN_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "tweets_clean.csv")

# Pour la détection de langue (facultative)
DetectorFactory.seed = 0

# ------------------------------------------------------------------------------
# Session Bluesky → Bearer Token
# ------------------------------------------------------------------------------
def create_session(handle: str, password: str) -> str | None:
    """
    Authentifie auprès de l'API Bluesky et renvoie un Bearer Token (accessJwt).
    """
    url = "https://bsky.social/xrpc/com.atproto.server.createSession"
    payload = {"identifier": handle, "password": password}
    resp = requests.post(url, json=payload)
    if resp.status_code == 200:
        return resp.json().get("accessJwt")
    print(f"❌ createSession failed [{resp.status_code}]: {resp.text}")
    return None

# ------------------------------------------------------------------------------
# Récupération de la timeline
# ------------------------------------------------------------------------------
def get_timeline(token: str, limit: int = TIMELINE_LIMIT) -> dict | None:
    """
    Récupère la timeline publique de l'utilisateur authentifié.
    """
    url = "https://bsky.social/xrpc/app.bsky.feed.getTimeline"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"limit": limit}
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code == 200:
        return resp.json()
    print(f"❌ getTimeline failed [{resp.status_code}]: {resp.text}")
    return None

# ------------------------------------------------------------------------------
# Extraction des tweets
# ------------------------------------------------------------------------------
def extract_tweets(timeline_data: dict) -> list[dict]:
    """
    Transforme la réponse JSON de getTimeline en une liste de dicts simplifiés.
    """
    tweets = []
    for item in timeline_data.get("feed", []):
        post   = item.get("post", {})
        record = post.get("record", {})
        author = post.get("author", {})
        tweets.append({
            "uri": post.get("uri", ""),
            "handle": author.get("handle", ""),
            "text": record.get("text", ""),
            "createdAt": record.get("createdAt", "")
        })
    return tweets

# ------------------------------------------------------------------------------
# Nettoyage et lemmatisation spaCy
# ------------------------------------------------------------------------------
# 1) Installer et télécharger le modèle français (une seule fois):
#    $ python -m spacy download fr_core_news_sm
nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])

def clean_text(text: str) -> str:
    """
    Nettoie et lemmatise un texte avec spaCy:
      - Supprime URLs, mentions, hashtags, chiffres, emojis
      - Conserve uniquement les tokens alphabetiques non-stop
      - Retourne la forme lemma.lower() jointe par des espaces
    """
    # 1. Nettoyage rapide regex avant spaCy
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\x00-\x7F]", "", text)  # emojis & non-ASCII
    # 2. Pipeline spaCy
    doc = nlp(text)
    lemmas = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop
    ]
    return " ".join(lemmas)

# ------------------------------------------------------------------------------
# Détection de langue (facultative)
# ------------------------------------------------------------------------------
def detect_language(text: str) -> str:
    """
    Renvoie le code de langue (ex: 'fr', 'en') ou 'unknown'.
    """
    try:
        return detect(text)
    except:
        return "unknown"

# ------------------------------------------------------------------------------
# Sauvegarde CSV
# ------------------------------------------------------------------------------
def save_to_csv(tweets: list[dict], path: str) -> None:
    """
    Écrit une liste de tweets dans un CSV avec '|' comme séparateur et QUOTE_ALL.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["uri", "handle", "text", "createdAt", "lang", "record"],
            delimiter="|",
            quotechar='"',
            quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        for t in tweets:
            t["record"] = json.dumps(t, ensure_ascii=False)
            writer.writerow(t)

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    # 1) Authentification
    token = create_session(HANDLE_OR_EMAIL, PASSWORD)
    if not token:
        return

    # 2) Récupération
    data = get_timeline(token)
    if not data:
        return

    # 3) Extraction
    raw_tweets = extract_tweets(data)
    print(f"Tweets bruts récupérés : {len(raw_tweets)}")

    # 4) Sauvegarde brute
    save_to_csv(
        [ {**t, "lang": detect_language(t["text"])} for t in raw_tweets ],
        RAW_CSV_PATH
    )
    print(f"✅ {len(raw_tweets)} tweets bruts sauvegardés dans {RAW_CSV_PATH}")

    # 5) Nettoyage & filtration (fr/en)
    cleaned = []
    for t in raw_tweets:
        cleaned_text = clean_text(t["text"])
        lang = detect_language(cleaned_text)
        if lang in ("fr", "en"):
            cleaned.append({
                **t,
                "text": cleaned_text,
                "lang": lang
            })
    print(f"Tweets filtrés (fr/en) : {len(cleaned)}")

    # 6) Sauvegarde nettoyée
    save_to_csv(cleaned, CLEAN_CSV_PATH)
    print(f"✅ {len(cleaned)} tweets propres sauvegardés dans {CLEAN_CSV_PATH}")

    # 7) Aperçu
    for t in cleaned:
        print(f"({t['handle']}) {t['createdAt']} [{t['lang']}] : {t['text']}\n")

if __name__ == "__main__":
    main()
