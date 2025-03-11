from api.create_session import create_session
from api.get_timeline import get_timeline
from api.extract_tweets import extract_tweets
from api.cleaning import clean_text


def main():
    # 1. Obtenir le token
    token = create_session("ton.handle.bsky.social", "ton_mot_de_passe")

    # 2. Récupérer la timeline
    timeline_data = get_timeline(token, limit=5)

    # 3. Extraire et nettoyer
    if timeline_data:
        tweets = extract_tweets(timeline_data)
        for t in tweets:
            t["text"] = clean_text(t["text"])

        # 4. Afficher un aperçu
        for tweet in tweets:
            print(f"({tweet['handle']}) {tweet['createdAt']} : {tweet['text']}\n")


if __name__ == "__main__":
    main()
