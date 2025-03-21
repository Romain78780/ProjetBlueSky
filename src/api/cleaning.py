import re
import nltk
from nltk.corpus import stopwords

# Vérifier et télécharger les stopwords si nécessaire
try:
    stop_words = set(stopwords.words("french"))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words("french"))

# Assure-toi d'avoir téléchargé les stopwords (nltk.download("stopwords"))
stop_words = set(stopwords.words("french"))

def clean_text(text):
    # Supprime les URLs
    text = re.sub(r"http\S+", "", text)
    # Supprime les mentions
    text = re.sub(r"@\w+", "", text)
    # Supprime les caractères spéciaux et ponctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Met en minuscules
    text = text.lower()
    # Supprime les stopwords
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

# Exemple d'utilisation
if __name__ == "__main__":
    sample = "@user Voici un tweet avec une URL http://example.com et des emojis 😊!"
    print("Texte nettoyé :", clean_text(sample))
