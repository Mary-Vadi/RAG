# rag

**Retrieval-augmented detection of AI-generated text: a hands-on course.**

Can a computer tell whether a text was written by a person or by an AI model? This course builds a detector that answers that question in two ways, and then combines them:

1. a **classifier** that learns patterns from labelled examples, and
2. **retrieval**, which finds the most similar texts in a labelled collection and lets them vote.

You'll build every piece yourself, one short lesson at a time, starting from "what is an embedding?" and finishing with a working retrieval-augmented detector tested on real benchmark data. Along the way you'll learn the core skills behind every RAG system (embeddings, vector search, nearest neighbours) and one of the most important lessons in applied machine learning: a model that looks perfect on its own test set can fail on data from somewhere else.

The course is built on my own experiment in AI-generated text detection. The original full-size version is kept in [`experiments/`](experiments).

## Who is this for?

Students and anyone curious about NLP who can read basic Python. No machine-learning background is assumed: every idea is explained before it's used, with small examples you can run and change.

## The lessons

Each lesson is a Jupyter notebook that runs in Google Colab for free. Click a badge to open it.

| # | Lesson | You will learn | Runtime | |
|---|---|---|---|---|
| 1 | [What is retrieval-augmented detection?](lessons/01_what_is_retrieval_augmented_detection.ipynb) | the task, classifier vs retrieval, how this relates to RAG, the RAID and MAGE datasets | CPU | <a href="https://colab.research.google.com/github/Mary-Vadi/rag/blob/main/lessons/01_what_is_retrieval_augmented_detection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 2 | [Embeddings](lessons/02_embeddings.ipynb) | turning text into vectors, cosine similarity, what happens inside an embedding model | CPU or GPU | <a href="https://colab.research.google.com/github/Mary-Vadi/rag/blob/main/lessons/02_embeddings.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 3 | [Similarity search](lessons/03_similarity_search.ipynb) | nearest neighbours by brute force and with FAISS, reading a text's neighbours | CPU or GPU | <a href="https://colab.research.google.com/github/Mary-Vadi/rag/blob/main/lessons/03_similarity_search.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 4 | [Retrieval as a detector](lessons/04_retrieval_as_a_detector.ipynb) | neighbour voting, accuracy / precision / recall / F1, validation vs test sets | CPU or GPU | <a href="https://colab.research.google.com/github/Mary-Vadi/rag/blob/main/lessons/04_retrieval_as_a_detector.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 5 | [Training a classifier](lessons/05_training_a_classifier.ipynb) | fine-tuning ModernBERT, in-domain vs out-of-domain testing, domain shift | GPU | <a href="https://colab.research.google.com/github/Mary-Vadi/rag/blob/main/lessons/05_training_a_classifier.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| 6 | [Retrieval-augmented detection](lessons/06_retrieval_augmented_detection.ipynb) | blending the two, the decision window, tuning α, comparing embeddings | GPU | <a href="https://colab.research.google.com/github/Mary-Vadi/rag/blob/main/lessons/06_retrieval_augmented_detection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

Every lesson ends with **check-your-understanding questions** (answers hidden until you click) and **try-it-yourself exercises**. New terms are collected in the [glossary](GLOSSARY.md).

## How to take the course

**In Google Colab (easiest).** Click a badge above. Run the first code cell ("Setup"): it downloads this repository and installs what's needed. For lessons 5 and 6, switch to a GPU first: *Runtime → Change runtime type → T4 GPU*. The free tier is enough.

**On your own computer.**

```bash
git clone https://github.com/Mary-Vadi/rag.git
cd rag
pip install -r requirements.txt
jupyter lab lessons/
```

Lessons 5 and 6 fine-tune a transformer model, which really needs a GPU.

Take the lessons in order: each one builds on the ideas of the one before. Datasets and models are downloaded from the Hugging Face Hub the first time you use them.

## The big picture

```
                  ┌───────────────────────────┐
  new text ──┬──▶ │ classifier (lesson 5)     │ ── P(machine) ──┐
             │    └───────────────────────────┘                 ▼
             │    ┌───────────────────────────┐            ┌──────────────┐
             └──▶ │ retrieval (lessons 2 to 4)│ ── vote ──▶│    blend     │──▶ human or machine
                  │ similar labelled texts    │            │  (lesson 6)  │
                  └───────────────────────────┘            └──────────────┘
```

Both halves produce `P(machine)`, a number between 0 and 1. The final answer is a weighted average of the two, `α × classifier + (1 − α) × retrieval`, and anything at 0.5 or above counts as machine-generated.

### Is this RAG?

Not in the usual sense, and the difference matters. **RAG** (retrieval-augmented *generation*) retrieves documents and gives them to a large language model, which writes an answer. This course keeps the **retrieval** half and replaces generation with a vote, so **no LLM is needed**. The retrieval skills you learn (embeddings, vector indexes, nearest-neighbour search) are exactly the ones every RAG system is built on. Lesson 1 explains the difference in detail, and lesson 6 suggests how to add an LLM on top as a project.

## What's in the repository

```
lessons/        the six course notebooks
ragdetect/      the Python code the lessons build and reuse
tests/          automated tests for ragdetect (run with pytest)
experiments/    the original full-size experiment and its write-up
GLOSSARY.md     every technical term used in the course, in plain English
```

The `ragdetect` package is small and meant to be read. Each module matches a lesson:

| Module | What it does | Lesson |
|---|---|---|
| `data.py` | loads RAID and MAGE samples in one common format (0 = human, 1 = machine) | 1 |
| `embeddings.py` | `Embedder`: text → unit-length vector, with mean pooling | 2 |
| `retrieval.py` | `NeighbourIndex` (FAISS search) and `RetrievalDetector` (neighbour vote) | 3, 4 |
| `evaluation.py` | accuracy, precision, recall, F1, confusion matrices | 4 |
| `classifier.py` | fine-tune a transformer and predict `P(machine)` | 5 |
| `augment.py` | `blend`, `decision_window` and `tune_alpha` | 6 |

## The original experiment in one paragraph

A ModernBERT classifier fine-tuned on RAID scored **98.8%** accuracy on held-out RAID texts but only **53.8%** on MAGE, flagging nearly 9 out of 10 human MAGE texts as AI. Adding retrieval with the blend weight fixed at α = 0.7 raised that to just **54.2%**. The course investigates why (spot checks suggest the training data was almost entirely scientific abstracts, and a confident classifier leaves retrieval very little room to act) and what to do about it. The full write-up is in [`experiments/README.md`](experiments/README.md).

## Running the tests

The tests use tiny, randomly initialised models and made-up data, so they run offline in a few seconds:

```bash
pip install pytest
pytest
```

## Datasets and model

- **RAID**: Dugan et al., *RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors*, ACL 2024. [liamdugan/raid](https://huggingface.co/datasets/liamdugan/raid)
- **MAGE**: Li et al., *MAGE: Machine-generated Text Detection in the Wild*, ACL 2024. [yaful/MAGE](https://huggingface.co/datasets/yaful/MAGE)
- **ModernBERT**: Warner et al., 2024. [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)
- **MiniLM** sentence embeddings: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **FAISS** for similarity search: [facebookresearch/faiss](https://github.com/facebookresearch/faiss)

Please check each dataset's licence before reusing it.

## About

I'm Maryam Vadikheil, an MSc Computer Science student at the University of Salford, working on AI-generated text detection. If something in the course is unclear or broken, please [open an issue](https://github.com/Mary-Vadi/rag/issues). Questions and ideas are welcome too, here or on [LinkedIn](https://www.linkedin.com/in/maryam-vadikheil).
