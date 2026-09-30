# The original experiment

This folder holds the full-size experiment the course is built on: a ModernBERT classifier trained on RAID, tested on MAGE, with and without a retrieval step on top.

<a href="https://colab.research.google.com/github/Mary-Vadi/rag/blob/main/experiments/original_experiment.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

```
original_experiment.ipynb   the whole pipeline in one Colab notebook, with its saved outputs
figures/                    confusion matrices from the MAGE evaluation
```

The notebook is self-contained: one of its early cells writes its own small `src/` package into the Colab filesystem. It is kept exactly as it was run, so the numbers below can be checked against its saved outputs. The lessons in [`../lessons`](../lessons) cover the same ideas step by step, on smaller samples.

## How it works

1. **Data.** The first 10% of RAID's training file and a random 10% of MAGE's train and test data. Texts under 50 characters are dropped, and each set is capped and balanced between human and machine as far as the data allows. Labels are 0 for human and 1 for machine; MAGE's labels are flipped on loading because it uses the opposite convention.
2. **Classifier.** `answerdotai/ModernBERT-base` fine-tuned for two epochs on RAID, with a stratified 80/10/10 train/validation/test split. The checkpoint with the best validation accuracy is kept.
3. **Retrieval index.** The fine-tuned encoder embeds the MAGE training texts (mean pooling, L2-normalised, 768 dimensions) into an exact FAISS inner-product index.
4. **Prediction.** Each MAGE test text's 7 nearest neighbours vote (weighted by `exp(similarity)`, which is almost a plain vote), and the vote is blended with the classifier: `final = 0.7 × classifier + 0.3 × retrieval`. Anything at 0.5 or above is called machine-generated.

The retrieval step uses labelled MAGE data from MAGE's train split, which never overlaps the test data. So the retrieval result is not zero-shot: it answers the question "if you have labelled examples from a new domain, can you use them without retraining?"

## Data

| Set | Source | Texts |
|---|---|---|
| Train | RAID | 35,540 |
| Validation | RAID | 4,442 |
| Internal test | RAID | 4,443 |
| Retrieval pool | MAGE train | 19,401 (9,401 human, 10,000 machine) |
| External test | MAGE test | 6,021 (3,044 human, 2,977 machine) |

The RAID sample (44,425 texts) is 19,425 human and 25,000 machine. Human texts are only about 3.5% of the RAID slice used, so the human side ran out before reaching the cap.

Two details about the data turned out to matter:

- **RAID's training file is sorted by domain.** Spot checks of rows 0, 20,000, 300,000 and 560,000 all show scientific abstracts, so the "first 10%" was very likely almost all abstracts. The slice also kept texts with adversarial attacks.
- **MAGE's `test` split on the Hugging Face Hub also includes its two out-of-distribution files**, because they're picked up automatically by name. The course's loaders read `test.csv` on its own to keep the sets clean.

## Results

Single run, seed 42, on a Colab T4 GPU (about 52 minutes of training). Precision, recall and F1 treat "machine" as the positive class.

| Evaluation | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| RAID internal test | 98.76 | 99.20 | 98.60 | 98.90 |
| MAGE, classifier only | 53.78 | 51.73 | 97.62 | 67.62 |
| MAGE, with retrieval | 54.16 | 51.94 | 97.72 | 67.82 |

<p align="center">
  <img src="figures/confusion_mage_baseline.png" width="45%" alt="MAGE confusion matrix, ModernBERT only">
  <img src="figures/confusion_mage_rag.png" width="45%" alt="MAGE confusion matrix, retrieval-augmented">
</p>

## What the results say

**The classifier learned RAID abstracts, not AI text in general.** Nearly 99% accuracy on RAID's held-out texts drops to about 54% on MAGE, barely better than guessing. It still catches almost every machine-written MAGE text (2,906 of 2,977), but it also flags 2,712 of the 3,044 human texts as machine. Given that its training data was very likely almost all scientific abstracts, this is domain shift in its clearest form. [Lesson 5](../lessons/05_training_a_classifier.ipynb) reproduces it on a small scale.

**Retrieval helped, but only slightly:** a net 23 more texts right (20 human, 3 machine), about +0.4 points. With α = 0.7, even if all seven neighbours say "human", a text only flips to human when the classifier's machine probability is below about 0.71. If the classifier is confidently wrong on most human MAGE texts, retrieval never gets a say. The probability distribution wasn't saved in this run, so that remains the likely explanation rather than a measured one. [Lesson 6](../lessons/06_retrieval_augmented_detection.ipynb) measures it and tunes α on validation data instead of fixing it.

**The embeddings probably carry the same bias.** The retrieval vectors come from the RAID-trained model, so the neighbourhoods they build on MAGE may be organised around RAID-specific cues. Lesson 6 compares them with a general-purpose embedding model.

## Next steps

- Sample RAID across all of its domains rather than taking the first 10% of the file.
- Tune α and the decision threshold on a held-out slice of MAGE, and calibrate the classifier's probabilities.
- Try other embedding models for retrieval, and weight closer neighbours more strongly.
- Evaluate on MAGE's out-of-distribution test files separately.
