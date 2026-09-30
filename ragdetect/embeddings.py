"""Turning text into vectors (lesson 2)."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Maps each text to a single vector of length 1.

    Works with any Hugging Face encoder, including a classifier you have
    fine-tuned yourself. The model gives one vector per token; we average them
    (ignoring padding) to get one vector per text. This is called mean pooling.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, max_length: int = 256, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.max_length = max_length

    @property
    def dim(self) -> int:
        return self.model.config.hidden_size

    @torch.inference_mode()
    def encode(self, texts, batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dim), one unit-length row per text."""
        texts = list(texts)
        starts = range(0, len(texts), batch_size)
        if show_progress:
            from tqdm.auto import tqdm
            starts = tqdm(starts, desc="Embedding", unit="batch")

        chunks = []
        for start in starts:
            batch = self.tokenizer(
                texts[start:start + batch_size],
                padding=True, truncation=True, max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            token_vectors = self.model(**batch).last_hidden_state          # (batch, tokens, dim)
            mask = batch["attention_mask"].unsqueeze(-1).to(token_vectors.dtype)
            mean = (token_vectors * mask).sum(1) / mask.sum(1).clamp(min=1)  # (batch, dim)
            unit = torch.nn.functional.normalize(mean, dim=1)
            chunks.append(unit.float().cpu().numpy())

        if not chunks:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.concatenate(chunks)
