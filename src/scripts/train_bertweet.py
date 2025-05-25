import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import torch

# ------------------------------------------------------------------------------
# Chemins d’accès (adaptés à ton arborescence)
# ------------------------------------------------------------------------------
PROJECT_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FACT_TABLE_PATH = os.path.join(PROJECT_ROOT, "src", "data", "fact_table.csv")
OUTPUT_DIR      = os.path.join(PROJECT_ROOT, "src", "models", "bertweet-fake-news")

def main():
    # 1) Charger la table de faits
    df = pd.read_csv(FACT_TABLE_PATH, sep="|", quotechar='"')
    df = df[["text", "label_bin"]].rename(columns={"text": "input_text", "label_bin": "label"})

    # 2) Train/Test split via sklearn (stratifié)
    train_df, eval_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )
    train_df = train_df.reset_index(drop=True)
    eval_df  = eval_df.reset_index(drop=True)

    # 3) Conversion en HuggingFace Dataset
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    eval_ds  = Dataset.from_pandas(eval_df,  preserve_index=False)

    # 4) Tokenisation avec BERTweet (max_length réduit à 96)
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=True)
    def tokenize_batch(batch):
        return tokenizer(
            batch["input_text"],
            padding="max_length",
            truncation=True,
            max_length=96
        )
    train_ds = train_ds.map(tokenize_batch, batched=True)
    eval_ds  = eval_ds.map(tokenize_batch, batched=True)

    # 5) Préparation pour PyTorch
    train_ds = train_ds.rename_column("label", "labels")
    eval_ds  = eval_ds.rename_column("label", "labels")
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # 6) Charger le modèle pour classification binaire
    model = AutoModelForSequenceClassification.from_pretrained(
        "vinai/bertweet-base",
        num_labels=2
    )

    # 7) Configurer les arguments de training
    total_steps = len(train_ds) // 8 + 1
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        fp16=True,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=total_steps,
        save_steps=total_steps,
        save_total_limit=2,
        learning_rate=2e-5
    )

    # 8) Instancier le Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer
    )

    # 9) Fine-tuning
    print("🔄 Fine-tuning de BERTweet sur la table de faits…")
    trainer.train()

    # 10) Évaluation finale
    print("🔄 Évaluation sur le jeu de test…")
    metrics = trainer.evaluate(eval_ds)
    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 11) Sauvegarde finale du modèle
    trainer.save_model(OUTPUT_DIR)
    print(f"\n✅ Modèle BERTweet saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
