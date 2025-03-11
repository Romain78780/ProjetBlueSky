import requests


def create_session(handle, password):
    """
    Crée une session Bluesky et renvoie le Bearer Token (accessJwt).

    :param handle: Ton handle Bluesky (ex: "ton.handle.bsky.social") ou ton e-mail
    :param password: Ton mot de passe Bluesky
    :return: accessJwt (string) si succès, None si erreur
    """
    url = "https://bsky.social/xrpc/com.atproto.server.createSession"
    payload = {
        "identifier": "romain.vignard@supdevinci-edu.fr",  # handle ou email
        "password": "ProjetBlueSkyM1"
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        data = response.json()
        return data.get("accessJwt")  # On récupère le token d'accès
    else:
        print(f"❌ Erreur {response.status_code} : {response.text}")
        return None


if __name__ == "__main__":
    # Mets ici tes identifiants locaux (ne les partage pas publiquement !)
    my_handle_or_email = "romain.vignard@supdevinci-edu.fr"  # Ou ton e-mail
    my_password = "ProjetBlueSkyM1"

    token = create_session(my_handle_or_email, my_password)
    if token:
        print("Bearer Token récupéré avec succès !")
        print("Access JWT :", token)
    else:
        print("Impossible de récupérer le Bearer Token.")
