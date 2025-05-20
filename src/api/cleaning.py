#!/usr/bin/env python3
# src/api/cleaning.py

import re
import spacy
from langdetect import detect, DetectorFactory

# Pour que la détection de langue soit déterministe
DetectorFactory.seed = 0

_nlp = None
def load_spacy_model():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
        except OSError:
            from spacy.cli import download as spacy_download
            print("Modèle fr_core_news_sm manquant, téléchargement…")
            spacy_download("fr_core_news_sm")
            _nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
    return _nlp

def clean_text(text: str) -> str:
    """Nettoie + lemmatise un texte (URLs, mentions, chiffres, stop-words, emojis…)."""
    nlp = load_spacy_model()
    t = text.lower()
    t = re.sub(r"http\S+|www\.\S+", "", t)
    t = re.sub(r"[@#]\w+", "", t)
    t = re.sub(r"\d+", "", t)
    t = re.sub(r"[^\x00-\x7F]", "", t)
    doc = nlp(t)
    return " ".join(tok.lemma_.lower() for tok in doc if tok.is_alpha and not tok.is_stop)

def detect_language(text: str) -> str:
    """Retourne le code langue (fr, en, etc.) ou 'unknown'."""
    try:
        return detect(text)
    except:
        return "unknown"

# Liste de mots-clés permettant d'identifier un tweet à caractère politique
POLITICAL_KEYWORDS = [
    "élection","vote","politique","président","gouvernement",
    "parlement","loi","ministre","démocratie","réforme",
    "senat","député","campagne","budget","impôt","manifestation",
    "justice","opposition","majorité","parti","droit","affaire"
]

def is_political(text: str) -> bool:
    """Vrai si le texte contient au moins un mot-clé politique."""
    txt = text.lower()
    return any(kw in txt for kw in POLITICAL_KEYWORDS)
