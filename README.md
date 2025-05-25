# **ProjetBlueSky**

Un pipeline complet pour :

* Récupérer des posts de l’API Bluesky (timeline ou recherche par mot-clé)

* Nettoyer et filtrer les tweets (FR/EN)

* Classifier les tweets en vrais vs fake news avec deux modèles :

* TF–IDF + LogisticRegression

* BERTweet fine-tuné

* Valider & calibrer la qualité du modèle (k-fold CV, CalibratedClassifierCV)

* Sortir deux CSV : tweets_all.csv (brut) et tweets_clean.csv (filtré + prédictions)

## Prérequis

* Python 3.8+

* (Optionnel) GPU CUDA pour BERTweet

* Clé d’accès Bluesky + identifiants dans .env

## Installation

1. Cloner le projet

git clone https://github.com/Romain78780/ProjetBlueSky.git
2. Aller dans le dossier du projet

cd ProjetBlueSky
3. Installer les dépendances

pip install --upgrade pip
pip install -r requirements.txt
4. Générer et remplir le fichier .env

Créer le fichier .env dans ProjetBlueSky\.env
Puis renseigner :


API_TOKEN=eyJ0eXAiOiJhdCtqd3QiLCJhbGciOiJFUzI1NksifQ…
BSKY_HANDLE=romain.vignard@supdevinci-edu.fr
BSKY_PASSWORD=ProjetBlueSkyM1
5. Télécharger les ressources NLP


python -m spacy download fr_core_news_sm

python - <<EOF
import nltk
nltk.download('stopwords')
EOF

## Table de faits LIAR

Le fichier data/processed/fact_table.csv (12 837 lignes) est fourni.
Pour le régénérer :


python src/scripts/build_fact_table.py

## Entraînement

6. Modèle TF–IDF + LogisticRegression


python src/scripts/train_model.py
→ models/fake_news_clf.joblib

7. Fine-tuning HuggingFace BERTweet (très lourd)

python src/scripts/train_bertweet.py
→ models/bertweet-fake-news/checkpoint-*

8. Validation & calibration (k-fold CV + CalibratedClassifierCV)


python src/scripts/validate_and_calibrate.py
→ models/fake_news_clf_calibrated.pkl

## Pipeline complet

9. Récupération, filtrage & classification (mot-clé "politics" par défaut)

python src/main.py
Génère :

* data/processed/tweets_all.csv

* data/processed/tweets_clean.csv

## Contact
