import pandas as pd
import pytest
from datasets import Dataset

import ragdetect.data as data


def mage_rows(n_human, n_machine):
    rows = [{"text": f"Human written post number {i}, long enough to keep.", "label": 1, "src": "cmv_human"}
            for i in range(n_human)]
    rows += [{"text": f"Machine generated passage number {i}, also long enough.", "label": 0, "src": "xsum_gpt"}
             for i in range(n_machine)]
    rows.append({"text": "too short", "label": 1, "src": "cmv_human"})
    return rows


def raid_row(model, attack="none", text="A generated passage that is comfortably over fifty characters."):
    return {"model": model, "attack": attack, "generation": text, "domain": "abstracts"}


def test_load_mage_flips_labels_filters_and_balances(monkeypatch):
    calls = []

    def fake_load_dataset(name, data_files=None, split=None, **kwargs):
        calls.append((name, data_files, split))
        return Dataset.from_list(mage_rows(30, 30))

    monkeypatch.setattr(data, "load_dataset", fake_load_dataset)
    df = data.load_mage("validation", n=20)

    assert calls == [("yaful/MAGE", {"validation": "valid.csv"}, "validation")]
    assert list(df.columns) == ["text", "label", "source"]
    assert df["label"].value_counts().to_dict() == {data.HUMAN: 10, data.MACHINE: 10}
    assert (df.loc[df["source"] == "cmv_human", "label"] == data.HUMAN).all()
    assert (df.loc[df["source"] == "xsum_gpt", "label"] == data.MACHINE).all()
    assert df["text"].str.len().min() >= 50


def test_load_mage_rejects_unknown_split():
    with pytest.raises(ValueError):
        data.load_mage("dev")


def test_balanced_sample_keeps_all_of_a_scarce_label():
    df = pd.DataFrame({"text": ["x"] * 53, "label": [0] * 3 + [1] * 50})
    sample = data.balanced_sample(df, n=20)
    assert sample["label"].value_counts().to_dict() == {1: 10, 0: 3}


def test_balanced_sample_without_n_keeps_everything():
    df = pd.DataFrame({"text": list("abcde"), "label": [0, 1, 0, 1, 1]})
    assert sorted(data.balanced_sample(df, n=None)["text"]) == list("abcde")


def test_load_raid_skips_attacks_short_texts_and_stops_at_max_rows(monkeypatch):
    stream = (
        [raid_row("human")] * 5
        + [raid_row("human", attack="homoglyph")] * 5
        + [raid_row("chatgpt", text="short")] * 5
        + [raid_row("chatgpt")] * 5
        + [raid_row("gpt4")] * 5  # beyond max_rows, never read
    )
    monkeypatch.setattr(data, "load_dataset", lambda *args, **kwargs: iter(stream))

    df = data.load_raid(n=100, max_rows=20)

    assert df["label"].value_counts().to_dict() == {data.HUMAN: 5, data.MACHINE: 5}
    assert set(df["source"]) == {"human", "chatgpt"}
    assert list(df.columns) == ["text", "label", "source", "domain"]


def test_load_raid_can_include_attacks(monkeypatch):
    stream = [raid_row("human"), raid_row("human", attack="homoglyph"), raid_row("mpt")]
    monkeypatch.setattr(data, "load_dataset", lambda *args, **kwargs: iter(stream))
    df = data.load_raid(n=None, include_attacks=True)
    assert len(df) == 3


def test_load_raid_explains_an_empty_sample(monkeypatch):
    monkeypatch.setattr(data, "load_dataset", lambda *args, **kwargs: iter([raid_row("human", attack="number")]))
    with pytest.raises(RuntimeError, match="max_rows"):
        data.load_raid()
