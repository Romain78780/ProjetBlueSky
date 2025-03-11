import csv


def extract_tweets(timeline_data):
    tweets_list = []
    feed = timeline_data.get("feed", [])
    for item in feed:
        post = item.get("post", {})
        record = post.get("record", {})
        author = post.get("author", {})

        tweet_info = {
            "uri": post.get("uri", ""),
            "handle": author.get("handle", "inconnu"),
            "text": record.get("text", ""),
            "createdAt": record.get("createdAt", "")
        }
        tweets_list.append(tweet_info)
    return tweets_list


# Exemple d'écriture dans un fichier CSV
def save_tweets_csv(tweets, filename="tweets.csv"):
    fieldnames = ["uri", "handle", "text", "createdAt"]
    with open(filename, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for tweet in tweets:
            writer.writerow(tweet)


if __name__ == "__main__":
    # Supposons que timeline_data est le JSON que tu as déjà récupéré
    timeline_data = {...}  # Remplace par tes données récupérées
    tweets = extract_tweets(timeline_data)
    save_tweets_csv(tweets)
    print(f"{len(tweets)} tweets extraits et sauvegardés dans tweets.csv")
