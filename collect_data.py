import os
from atproto import Client
from dotenv import load_dotenv

# loading variables
load_dotenv()

username =os.getenv("BLUESKY_USERNAME")
pwd = os.getenv("BLUESKY_PASSWORD")

client = Client()
client.login(username, pwd)

print("Cnx réussie", client)