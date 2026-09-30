"""Embedding and fine-tuning, run end to end on tiny random models."""

from pathlib import Path

import numpy as np
import pandas as pd

from ragdetect.classifier import predict_proba, train_classifier
from ragdetect.embeddings import Embedder
from tiny_models import synthetic_corpus


def test_embedder_returns_one_unit_vector_per_text(tiny_bert):
    embedder = Embedder(tiny_bert)
    vectors = embedder.encode(["first text", "second, rather longer text", "third"], batch_size=2)
    assert vectors.shape == (3, embedder.dim)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_padding_does_not_change_an_embedding(tiny_bert):
    embedder = Embedder(tiny_bert)
    alone = embedder.encode(["a short text"])
    padded = embedder.encode(["a short text", "a much longer text that forces padding " * 5])
    np.testing.assert_allclose(alone[0], padded[0], atol=1e-5)


def test_embedder_handles_no_texts(tiny_bert):
    assert Embedder(tiny_bert).encode([]).shape == (0, 32)


def test_train_predict_and_embed_with_the_finetuned_model(tiny_modernbert, tmp_path):
    texts, labels = synthetic_corpus(240, seed=1)
    df = pd.DataFrame({"text": texts, "label": labels})
    train_df, val_df = df.iloc[:200], df.iloc[200:]

    model_dir = train_classifier(
        train_df, val_df, output_dir=str(tmp_path / "clf"), model_name=tiny_modernbert,
        epochs=3, learning_rate=3e-3, batch_size=16, max_length=64,
    )

    assert not list(Path(model_dir).glob("checkpoint-*"))
    proba = predict_proba(model_dir, val_df["text"], max_length=64)
    assert proba.shape == (40,)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert ((proba >= 0.5) == val_df["label"].to_numpy()).mean() > 0.75

    # Lesson 6 reuses the fine-tuned classifier as an embedding model.
    vectors = Embedder(model_dir, max_length=64).encode(val_df["text"][:5])
    assert vectors.shape == (5, 32)
