# src/api/create_session.py

import os
import requests
from dotenv import load_dotenv

# Charge les variables d'environnement depuis .env
load_dotenv()

def create_session(handle: str = None, password: str = None) -> str | None:
    """
    Crée une session Bluesky et renvoie le Bearer Token (accessJwt).
    Si handle/password ne sont pas passés en argument, on les lit depuis .env.
    """
    # Priorité aux arguments, sinon on lit du .env
    identifier = handle or os.getenv("BSKY_HANDLE")
    pwd        = password or os.getenv("BSKY_PASSWORD")

    if not identifier or not pwd:
        print("❌ Veuillez définir BSKY_HANDLE et BSKY_PASSWORD dans votre .env ou les passer à la fonction")
        return None

    url = "https://bsky.social/xrpc/com.atproto.server.createSession"
    payload = {"identifier": identifier, "password": pwd}
    resp = requests.post(url, json=payload)

    if resp.status_code == 200:
        return resp.json().get("accessJwt")
    else:
        print(f"❌ createSession failed [{resp.status_code}]: {resp.text}")
        return None

if __name__ == "__main__":
    # Test rapide
    token = create_session()
    if token:
        print("✅ Bearer Token récupéré :", token)
    else:
        print("❌ Échec de la récupération du token")
