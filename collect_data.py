import os
from atproto import Client
from dotenv import load_dotenv

# loading variables
load_dotenv()

username =os.getenv("BLUESKY_USERNAME")
pwd = os.getenv("BLUESKY_PASSWORD")

client = Client()
client.login(username, pwd)

timeline = client.get_timeline()
#
#for item in timeline:
#   post = item.post
 #   print(f"{post.author.handle}:{post.record.text}")

# Récupérer les posts d'un utilisateur spécifique (remplace handle par l'identifiant)
handle = "mouhoub.bsky.social"
posts = client.repo.list_records(
    repo="handle.bsky.social",
    collection="app.bsky.feed.post"
)

# Afficher les posts récupérés
for post in posts.records:
    print(f"{post.author}: {post.text}")
