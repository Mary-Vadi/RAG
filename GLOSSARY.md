# Glossary

Every technical term used in the course, in plain English. The number in brackets is the lesson where it's introduced.

**Accuracy** (4). The share of all texts a detector labels correctly. Easy to read, but it hides *which* mistakes are being made.

**Adversarial attack** (1). A deliberate change to a text, such as misspellings, swapped letters or paraphrasing, made to fool a detector. RAID includes many.

**α, alpha** (6). The weight on the classifier when blending: `α × classifier + (1 − α) × retrieval`. α = 1 means only the classifier counts; α = 0 means only retrieval counts.

**Attention mask** (2). A list of 1s and 0s that tells a model which positions in a batch are real tokens (1) and which are padding (0).

**Batch** (2). A group of texts processed together. Faster than one at a time, but shorter texts must be padded to the same length.

**Binary classification** (1). Assigning each input one of exactly two labels, here human (0) or machine (1).

**Classification head** (5). A small layer added on top of a pre-trained model that turns its output into one score per label. It starts out random and is learned during fine-tuning.

**Classifier** (1). A model that learns patterns from labelled examples and then judges each new input on its own.

**Confusion matrix** (4). A 2 × 2 table of true labels (rows) against predicted labels (columns). It shows exactly which mistakes a detector makes.

**Cosine similarity** (2). How closely two vectors point in the same direction: 1 means identical direction, 0 means unrelated. For vectors of length 1 it equals the dot product.

**Decision window** (6). The range of classifier scores inside which retrieval can still change the final answer. Outside it, the classifier's decision wins whatever the neighbours say.

**Domain** (1). The kind of text: news, scientific abstracts, Reddit posts, recipes and so on.

**Domain shift** (5). When a model is used on a different kind of data from the data it was trained on. Performance often drops sharply.

**Dot product / inner product** (2). Multiply two vectors number by number and add up the results.

**Embedding** (2). A vector (list of numbers) that represents a text, chosen so that similar texts get similar vectors.

**Epoch** (5). One full pass of training over the whole training set.

**F1 score** (4). A single number that combines precision and recall (their harmonic mean). It is only high when both are high.

**FAISS** (3). A library from Meta for fast similarity search over large collections of vectors.

**Fine-tuning** (5). Taking a pre-trained model and training it a little further on a specific task with labelled data.

**Index** (3). A data structure that stores vectors so they can be searched quickly. `IndexFlatIP` is exact; approximate indexes (IVF, HNSW) trade a little accuracy for a lot of speed.

**In-domain / out-of-domain** (5). Testing on the same kind of data the model was trained on (in-domain) or a different kind (out-of-domain).

**k** (3). How many nearest neighbours to retrieve for each query.

**Label** (1). The correct answer for an example: 0 for human-written, 1 for machine-generated in this course.

**LLM, large language model** (1). A model such as GPT or Llama that generates text. RAG systems use one to write answers; this course's detector does not need one.

**Mean pooling** (2). Averaging a model's token vectors (ignoring padding) to get one vector for the whole text.

**Nearest neighbours** (3). The texts in a pool whose embeddings are most similar to a query's embedding.

**Normalising** (2). Scaling a vector to length 1 so that dot products become cosine similarities.

**Padding** (2). Filler tokens added to shorter texts so every text in a batch has the same length.

**PCA, principal component analysis** (2). A way to squash many dimensions down to two or three for plotting, keeping the directions along which the data varies most.

**Pool** (3). The collection of labelled texts that retrieval searches.

**Positive class** (4). The class we are trying to catch; "machine" in this course. Precision and recall are measured for it.

**Pre-trained model** (5). A model that has already learned general knowledge of language from a huge amount of text, ready to be fine-tuned.

**Precision** (4). Of the texts flagged as machine, the share that really are machine. Low precision means many false alarms.

**P(machine)** (1). A detector's estimated probability, between 0 and 1, that a text was machine-generated. 0.5 or above counts as "machine".

**RAG, retrieval-augmented generation** (1). Retrieve documents relevant to a question, then have an LLM generate an answer from them.

**Recall** (4). Of the machine texts, the share that were flagged. Low recall means AI text slips through.

**Retrieval** (1). Looking up the most similar items in a stored collection.

**Retrieval-augmented detection** (1). Blending a classifier's prediction with a vote from retrieved, labelled texts. The method taught in this course.

**Softmax** (4). A function that turns a list of scores into positive weights that add up to 1, giving higher scores bigger weights.

**Temperature** (4). A setting that controls how sharply softmax favours the highest scores. Small temperature: the top score dominates. Large temperature: weights become nearly equal.

**Test set** (4). Data kept aside to measure final performance, used once at the very end.

**Token** (2). A piece of text a model reads: a whole common word, or part of a rarer one.

**Tokenizer** (2). The tool that splits text into tokens and converts them into ID numbers.

**Training set** (5). The labelled examples a model learns from.

**Validation set** (4). Data used to choose settings (such as `k`, temperature or α) without touching the test set.

**Vector** (2). An ordered list of numbers, such as `[0.9, 0.1, 0.8]`.

**Zero-shot** (6). Working on a new domain without using any labelled examples from it. Our retrieval step is *not* zero-shot: it uses labelled MAGE texts.
