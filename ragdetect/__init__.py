"""The code behind the course, built up one lesson at a time.

    data        loading RAID and MAGE in one common format        (lesson 1)
    embeddings  turning text into vectors                          (lesson 2)
    retrieval   finding similar texts and letting them vote       (lessons 3 and 4)
    evaluation  accuracy, precision, recall, F1, confusion matrix (lesson 4)
    classifier  fine-tuning a transformer detector                (lesson 5)
    augment     combining the classifier with retrieval           (lesson 6)
"""

from ragdetect.data import HUMAN, LABEL_NAMES, MACHINE

__all__ = ["HUMAN", "MACHINE", "LABEL_NAMES"]
