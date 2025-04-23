import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def main():
    # 1) Charger la table de faits
    df = pd.read_csv(
        "data/processed/fact_table.csv",
        sep="|",
        quotechar='"'
    )

    X = df[[
        "text",
        "emotion_label",
        "n_tokens",
        "n_stopwords",
        "n_chars",
        "n_upper",
        "emotion_score"
    ]]
    y = df["label_bin"]

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3) Préprocesseur : TF-IDF, OHE, Scaling
    preprocessor = ColumnTransformer(
        [
            ("tfidf", TfidfVectorizer(max_features=5000), "text"),
            ("ohe_emotion", OneHotEncoder(handle_unknown="ignore"), ["emotion_label"]),
            ("scale_num", StandardScaler(), ["n_tokens", "n_stopwords", "n_chars", "n_upper", "emotion_score"]),
        ],
        remainder="drop"
    )

    # 4) Pipeline + LogisticRegression optimisé
    clf = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(
            solver="saga",
            max_iter=5000,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    # 5) Entraînement
    print("🔄 Entraînement du modèle…")
    clf.fit(X_train, y_train)

    # 6) Évaluation
    print("🔄 Évaluation sur le set de test…")
    y_pred = clf.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # 7) Sauvegarde
    import joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/fake_news_clf.joblib")
    print("\n✅ Modèle entraîné sauvegardé dans models/fake_news_clf.joblib")

if __name__ == "__main__":
    main()
