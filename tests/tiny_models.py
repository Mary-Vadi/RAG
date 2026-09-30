"""Tiny, randomly initialised stand-ins for the real models, so tests run offline in seconds."""

from __future__ import annotations

import random
from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
from transformers import BertConfig, BertModel, ModernBertConfig, ModernBertForMaskedLM, PreTrainedTokenizerFast

HUMAN_WORDS = "honestly reckon mate weird lol yeah kinda gonna tbh dunno stuff guess basically".split()
MACHINE_WORDS = "furthermore additionally comprehensive crucial delve notably significant overall moreover".split()
SHARED_WORDS = "the a of and to in is it that for on with as this was paper study results model data people time".split()


def synthetic_text(label: int, rng: random.Random) -> str:
    """A sentence that leans on human-ish (label 0) or machine-ish (label 1) words."""
    own = HUMAN_WORDS if label == 0 else MACHINE_WORDS
    words = [rng.choice(own if rng.random() < 0.4 else SHARED_WORDS) for _ in range(rng.randint(15, 40))]
    return " ".join(words).capitalize() + "."


def synthetic_corpus(n: int, seed: int = 0):
    rng = random.Random(seed)
    labels = [i % 2 for i in range(n)]
    return [synthetic_text(label, rng) for label in labels], labels


def _tokenizer(model_input_names) -> PreTrainedTokenizerFast:
    texts, _ = synthetic_corpus(400)
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(texts, trainers.WordPieceTrainer(
        vocab_size=300, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    ))
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]", model_input_names=model_input_names,
    )


def build_tiny_bert(path) -> str:
    """Stand-in for sentence-transformers/all-MiniLM-L6-v2 (a BERT encoder)."""
    path = Path(path)
    tokenizer = _tokenizer(["input_ids", "token_type_ids", "attention_mask"])
    config = BertConfig(
        vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=512, pad_token_id=tokenizer.pad_token_id,
    )
    BertModel(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


def build_tiny_modernbert(path) -> str:
    """Stand-in for answerdotai/ModernBERT-base (saved as a masked-LM checkpoint, like the real one)."""
    path = Path(path)
    tokenizer = _tokenizer(["input_ids", "attention_mask"])
    config = ModernBertConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, max_position_embeddings=512, pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id, sep_token_id=tokenizer.sep_token_id,
        bos_token_id=tokenizer.cls_token_id, eos_token_id=tokenizer.sep_token_id,
    )
    ModernBertForMaskedLM(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)
