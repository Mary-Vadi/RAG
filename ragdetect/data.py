"""Loading the two datasets used in the course (lesson 1).

Both loaders return a pandas DataFrame with the same columns:

    text    the passage itself
    label   0 = human-written, 1 = machine-generated
    source  who wrote it: a model name, or something marking it as human

so the rest of the course never needs to know which dataset a text came from.
"""

from __future__ import annotations

import pandas as pd
from datasets import load_dataset

HUMAN, MACHINE = 0, 1
LABEL_NAMES = ["human", "machine"]

MAGE_FILES = {"train": "train.csv", "validation": "valid.csv", "test": "test.csv"}


def balanced_sample(df: pd.DataFrame, n: int | None, seed: int = 42) -> pd.DataFrame:
    """Shuffle `df` and keep up to `n` rows, half of each label where possible.

    If one label has fewer than n / 2 rows, all of them are kept and the
    sample ends up smaller than `n` rather than lopsided.
    """
    if n is None:
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    per_label = n // 2
    parts = [
        group.sample(n=min(per_label, len(group)), random_state=seed)
        for _, group in df.groupby("label")
    ]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def load_mage(split: str = "test", n: int | None = 2000, seed: int = 42, min_chars: int = 50) -> pd.DataFrame:
    """A balanced sample of `n` texts from one MAGE split.

    `split` is "train", "validation" or "test". Only that split's file is
    downloaded. MAGE's train, validation and test files never share texts, so
    samples from different splits are safe to use side by side.
    """
    if split not in MAGE_FILES:
        raise ValueError(f"split must be one of {sorted(MAGE_FILES)}, got {split!r}")
    raw = load_dataset("yaful/MAGE", data_files={split: MAGE_FILES[split]}, split=split).to_pandas()
    df = pd.DataFrame({
        "text": raw["text"].astype(str),
        # MAGE uses 1 = human and 0 = machine, the opposite of this course.
        "label": (raw["label"] == 0).astype(int),
        "source": raw["src"].astype(str),
    })
    df = df[df["text"].str.strip().str.len() >= min_chars]
    return balanced_sample(df, n, seed)


def load_raid(
    n: int | None = 4000,
    seed: int = 42,
    min_chars: int = 50,
    include_attacks: bool = False,
    max_rows: int = 100_000,
) -> pd.DataFrame:
    """A balanced sample of up to `n` texts from the start of RAID's training file.

    The file is 11.8 GB, so rather than downloading all of it we stream the
    first `max_rows` rows and sample from those. RAID is sorted by domain, so
    those first rows are all scientific abstracts. That shortcut is also a
    trap, and lesson 5 is about why.

    By default texts that have been through one of RAID's adversarial attacks
    (misspellings, homoglyphs, paraphrasing and so on) are skipped.
    """
    stream = load_dataset("liamdugan/raid", split="train", streaming=True)
    rows = []
    for i, example in enumerate(stream):
        if i >= max_rows:
            break
        if not include_attacks and example["attack"] != "none":
            continue
        text = example["generation"] or ""
        if len(text.strip()) < min_chars:
            continue
        model = str(example["model"]).strip().lower()
        rows.append({
            "text": text,
            # RAID marks human-written texts with model == "human".
            "label": HUMAN if model == "human" else MACHINE,
            "source": model,
            "domain": example["domain"],
        })
    if not rows:
        raise RuntimeError(f"No usable RAID rows in the first {max_rows} rows; try a larger max_rows.")
    return balanced_sample(pd.DataFrame(rows), n, seed)
