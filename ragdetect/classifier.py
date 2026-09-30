"""Fine-tuning a transformer to label texts human or machine (lesson 5)."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from ragdetect.data import LABEL_NAMES

DEFAULT_MODEL = "answerdotai/ModernBERT-base"


def _accuracy(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": float((np.argmax(logits, axis=-1) == labels).mean())}


def train_classifier(
    train_df,
    val_df,
    output_dir: str = "models/classifier",
    model_name: str = DEFAULT_MODEL,
    epochs: int = 1,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    max_length: int = 256,
    seed: int = 42,
) -> str:
    """Fine-tune `model_name` on train_df, keep the epoch with the best validation accuracy.

    Both DataFrames need `text` and `label` columns. The finished model and
    tokenizer are saved to `output_dir`, whose path is returned.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=dict(enumerate(LABEL_NAMES)),
        label2id={name: i for i, name in enumerate(LABEL_NAMES)},
    )

    def to_dataset(df):
        ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
        return ds.map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
            batched=True, remove_columns=["text"],
        )

    train_ds, val_ds = to_dataset(train_df), to_dataset(val_df)
    total_steps = math.ceil(len(train_ds) / batch_size) * epochs

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_steps=max(1, total_steps // 10),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        logging_steps=25,
        report_to="none",
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_accuracy,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # The best weights are now saved at the top of output_dir; the
    # intermediate checkpoints are just using up disk space.
    for checkpoint in Path(output_dir).glob("checkpoint-*"):
        shutil.rmtree(checkpoint)
    return output_dir


@torch.inference_mode()
def predict_proba(model_dir: str, texts, batch_size: int = 32, max_length: int = 256) -> np.ndarray:
    """P(machine) for each text, from a classifier saved by train_classifier."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    texts = list(texts)
    probs = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start:start + batch_size],
            padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        ).to(device)
        logits = model(**batch).logits
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros(0, dtype=np.float32)
