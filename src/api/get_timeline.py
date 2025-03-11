import requests

from .create_session import create_session

def get_timeline(bearer_token, limit=10):
    """
    Récupère la timeline publique depuis l'API Bluesky en utilisant un Bearer Token.

    :param bearer_token: Jeton d'authentification (accessJwt) obtenu via create_session
    :param limit: Nombre de posts à récupérer
    :return: Un dictionnaire contenant les posts ou None en cas d'erreur.
    """
    API_URL = "https://bsky.social/xrpc/app.bsky.feed.getTimeline"
    headers = {
        "Authorization": f"Bearer {bearer_token}"
    }
    params = {
        "limit": limit
    }

    response = requests.get(API_URL, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()
    elif response.status_code == 401:
        print("❌ Erreur 401: Authentification requise. Le token est peut-être invalide ou expiré.")
    else:
        print(f"❌ Erreur {response.status_code}: {response.text}")
    return None


if __name__ == "__main__":
    my_handle_or_email = "ton.handle.bsky.social"  # Remplace par ton identifiant ou e-mail
    my_password = "TON_MOT_DE_PASSE"  # Remplace par ton mot de passe

    token = create_session(my_handle_or_email, my_password)
    if token:
        timeline_data = get_timeline(token, limit=5)
        if timeline_data:
            print("Timeline récupérée avec succès :")
            print(timeline_data)
    else:
        print("Impossible de récupérer le Bearer Token, vérifie tes identifiants.")
