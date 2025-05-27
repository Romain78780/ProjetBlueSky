#!/usr/bin/env python3
# src/dashboard/dashboard.py

import os
from pathlib import Path

import pandas as pd
import dash
from dash import html, dcc, dash_table
import plotly.express as px

# ── 1. Chemin vers le CSV ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CSV_PATH = BASE_DIR / "data" / "processed" / "tweets_clean.csv"

# ── 2. Chargement & parsing des dates ISO8601 ────────────────────────────────
df = pd.read_csv(CSV_PATH, sep="|", dtype={"createdAt": str})
df["createdAt"] = pd.to_datetime(
    df["createdAt"].str.replace(r"Z$", "+00:00", regex=True),
    utc=True,
    errors="coerce",
)
df = df.dropna(subset=["createdAt"])

# ── 3. Préparation des labels & agrégations ──────────────────────────────────
# mapping 0 → True, 1 → Fake
df["label"] = df["pred_label"].map({0: "True", 1: "Fake"})
# répartition True vs Fake
df_pred_counts = (
    df["label"]
    .value_counts()
    .reset_index(name="count")
    .rename(columns={"index": "label"})
)
# langues
lang_counts = (
    df["lang"]
    .value_counts()
    .reset_index(name="count")
    .rename(columns={"index": "lang"})
)
# top 10 tweets suspects
top_fake = df.sort_values("fake_score", ascending=False).head(10)

# ── 4. Initialisation de l’app Dash ────────────────────────────────────────────
app = dash.Dash(__name__)
app.title = "Dashboard Fake News Bluesky"

# ── 5. Layout ──────────────────────────────────────────────────────────────────
app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "auto", "padding": "20px"},
    children=[
        html.H1(
            "🧠 Dashboard Fake News Bluesky",
            style={"textAlign": "center", "marginBottom": "40px"},
        ),

        # 1) Histogramme des scores
        dcc.Graph(
            figure=px.histogram(
                df,
                x="fake_score",
                nbins=20,
                title="📊 Répartition des scores de suspicion",
            )
        ),

        # 2) Répartition True vs Fake
        dcc.Graph(
            figure=px.bar(
                df_pred_counts,
                x="label",
                y="count",
                title="🔍 Répartition Vrai vs Fake",
                text="count",
            ).update_traces(marker_color=["#2ca02c", "#d62728"])
        ),

        # 3) Camembert des langues
        dcc.Graph(
            figure=px.pie(
                lang_counts,
                names="lang",
                values="count",
                title="🌍 Répartition des langues",
            )
        ),

        # 4) Top 10 tweets suspects
        html.H2("🧪 Top 10 des tweets suspects", style={"marginTop": "40px"}),
        dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in ["handle", "text", "fake_score"]],
            data=top_fake.to_dict("records"),
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "whiteSpace": "normal"},
            style_header={"backgroundColor": "#f5f5f5", "fontWeight": "bold"},
            page_size=10,
        ),
    ],
)

# ── 6. Lancement du serveur ─────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
