import csv
import json
from langdetect import detect, DetectorFactory
from api.create_session import create_session
from api.get_timeline import get_timeline
from api.extract_tweets import extract_tweets
from api.cleaning import clean_text

# Pour éviter les variations aléatoires dans la détection de langue
DetectorFactory.seed = 0


def detect_language(text):
    """Détecte la langue d'un texte et retourne son code (fr, en, es, etc.)."""
    try:
        return detect(text)
    except:
        return "unknown"  # Si la détection échoue


def main():
    # 1. Obtenir le token
    token = create_session("ton.handle.bsky.social", "ton_mot_de_passe")

    # 2. Récupérer la timeline
    timeline_data = get_timeline(token, limit=10)  # On peut augmenter la limite

    # 3. Extraire, nettoyer, détecter la langue et filtrer
    if timeline_data:
        tweets = extract_tweets(timeline_data)
        filtered_tweets = []  # Initialisation correcte de filtered_tweets

        for t in tweets:
            t["text"] = clean_text(t["text"])
            t["lang"] = detect_language(t["text"])  # Détection de langue

            # On garde uniquement les tweets en français ou anglais
            if t["lang"] in ["fr", "en"]:
                filtered_tweets.append(t)  # Ajout des tweets filtrés

        # 4. Sauvegarde dans un fichier CSV
        save_to_csv(filtered_tweets, "tweets_filtered.csv")

        # 5. Afficher un aperçu
        for tweet in filtered_tweets:
            print(f"({tweet['handle']}) {tweet['createdAt']} [{tweet['lang']}] : {tweet['text']}\n")


def save_to_csv(tweets, filename):
    """Sauvegarde les tweets filtrés dans un fichier CSV."""
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["uri", "handle", "text", "createdAt", "lang", "record"])
        writer.writeheader()

        for tweet in tweets:
            tweet["record"] = json.dumps(tweet, ensure_ascii=False)  # Stocke tout l'objet en JSON
            writer.writerow(tweet)


if __name__ == "__main__":
    main()
