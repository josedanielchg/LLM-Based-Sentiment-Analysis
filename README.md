# Text Classification — Sentiment Analysis (MI201 Project 3)

Sentiment classification system for **short English texts (tweets)** with **3 classes** (Negative / Neutral / Positive). This project benchmarks **classical ML pipelines** (BoW/TF-IDF + linear/ensemble models), **MLP baselines**, **BERT-based encoders**, **LLM few-shot prompting**, and a **parameter-efficient fine-tuning (LoRA) approach** that achieves the best overall performance.

---

## Table of Contents

- [Project Summary](#project-summary)
- [Dataset](#dataset)
- [Evaluation Protocol](#evaluation-protocol)
- [Q0–Q1 — Classical Machine Learning Baselines](#q0q1--classical-machine-learning-baselines)
  - [Representations](#representations)
  - [Classifiers](#classifiers)
  - [Hyperparameter Search and Training Cost](#hyperparameter-search-and-training-cost)
- [Q2–Q3 — MLP Models on Vectorized Text](#q2q3--mlp-models-on-vectorized-text)
  - [Architectures and Training Setup](#architectures-and-training-setup)
  - [Generalization Behavior (Train–Test Gap)](#generalization-behavior-traintest-gap)
- [Q4 — Comparison with a Large Language Model (Few-Shot)](#q4--comparison-with-a-large-language-model-fewshot)
- [Q5 — BERT Embeddings vs. Raw Vectorizations](#q5--bert-embeddings-vs-raw-vectorizations)
- [Q6–Q7 — Transformer Fine-Tuning with LoRA (Best Model)](#q6q7--transformer-finetuning-with-lora-best-model)
  - [LoRA Configuration](#lora-configuration)
  - [Results and Error Profile](#results-and-error-profile)
- [Results Overview](#results-overview)
- [Repository Usage](#repository-usage)
  - [Environment](#environment)
  - [Artifacts and Output Conventions](#artifacts-and-output-conventions)
  - [Reproducibility Checklist](#reproducibility-checklist)
- [Key Takeaways](#key-takeaways)
- [Authors](#authors)

---

## Project Summary

Sentiment analysis of short texts is a practical NLP task for extracting public perception and emotional signals from social platforms. In this project we:

1. **Preprocess** and analyze a tweet-based dataset.
2. Train **classical pipelines** (vectorizer + classifier) with systematic hyperparameter search.
3. Train **MLP** classifiers on sparse vectorizations and on BERT embeddings.
4. Benchmark against a **few-shot LLM** inference approach (no weight updates).
5. Achieve the best performance with **BERT + LoRA fine-tuning**, adapting the encoder efficiently with low-rank updates.

---

## Dataset

- **Domain:** short English texts (tweets).
- **Task:** **3-class sentiment classification** (Negative, Neutral, Positive), mapped to numeric labels **0/1/2**.
- **Feature selection rationale:** the **processed text** (after stopword removal) is the primary training signal; other metadata (e.g., age ranges / country fields) can be non-informative or risk inducing spurious patterns, so the experiments focus on text as the main explanatory variable.

---

## Evaluation Protocol

We report:
- **Accuracy**
- **Macro-Recall**
- **Macro-F1**

Macro metrics are emphasized because they are more informative under class imbalance and prevent the dominant class from masking failures on minority classes.

All comparisons are performed consistently on the same test-set evaluation protocol.

---

## Q0–Q1 — Classical Machine Learning Baselines

Classical baselines are built as end-to-end **pipelines**:

> `Text → Vectorizer → Classifier → Prediction`

### Representations

We compare three standard vectorization schemes:

1. **BoW (Bag-of-Words)**  
   Sparse count-based features. Strong baseline for short texts.

2. **TF-IDF (word-level)**  
   Word n-grams weighted by inverse document frequency.

3. **TF-IDF (character-level)**  
   Character n-grams that better capture:
   - spelling noise and abbreviations,
   - informal writing and elongated words,
   - hashtags and sub-lexical sentiment cues.

### Classifiers

We evaluate standard classifiers commonly used for text classification:

- **Multinomial Naive Bayes (MNB)**
- **Logistic Regression (LR)**
- **Linear SVM**
- **Random Forest (RF)**

### Hyperparameter Search and Training Cost

A broad **GridSearchCV** is used to explore vectorizer + classifier configurations. For transparency and auditability, each experiment stores a **CSV** with the full candidate set, including `mean_fit_time`.

A compute-cost estimate is derived from the accumulated mean training times and the number of CV splits. With parallelization (`n_jobs=-1`), wall-clock time is reduced proportionally to the number of effective cores.

---

## Q2–Q3 — MLP Models on Vectorized Text

In addition to classical linear/ensemble baselines, we evaluate whether introducing non-linearity via MLPs improves performance over high-dimensional sparse inputs.

### Architectures and Training Setup

We train multiple MLP families, aligned to the representation:

- **Sparse vectorizations** (BoW, Word TF-IDF, Char TF-IDF): deeper/wider MLP variants.
- **BERT embeddings** (768-d dense vectors): smaller heads due to the already-compressed representation.

Training setup (consistent across MLP variants):
- Loss: **Cross-entropy**
- Optimizer: **Adam**
- Learning rate: **1e-4**
- Max epochs: **50**
- Early stopping criterion: **1e-4**
- Dropout inserted between hidden layers to control overfitting

### Generalization Behavior (Train–Test Gap)

A key observation: MLPs on sparse TF-IDF/BoW often reach very high training scores (sometimes close to 1.0) even with relatively small networks, revealing a strong **memorization capacity** in sparse, high-dimensional regimes. This does not guarantee proportional test improvements, so generalization control (dropout, early stopping, capacity constraints) is essential.

---

## Q4 — Comparison with a Large Language Model (Few-Shot)

We benchmark a generalist decoder-only LLM in **inference mode**:

- Model: **gemma-3-4b-it**
- Method: **few-shot prompting via API** (no parameter updates)
- Prompting design:
  - explicit instruction to act as sentiment expert,
  - **3 labeled examples** (one per class),
  - constrained outputs to single-word labels mapped to 0/1/2
- Evaluation performed on a **subset of 1,000 test samples** due to rate/latency constraints.

This comparison tests whether a large general-purpose model can rival a smaller encoder specialized for the dataset.

---

## Q5 — BERT Embeddings vs. Raw Vectorizations

BERT provides **contextualized, bidirectional** representations and outputs a **768-dimensional embedding** per text (after pooling). This dramatically reduces classifier input dimensionality compared to sparse TF-IDF, which can reduce parameter count and overfitting risk when training downstream MLP heads.

A quantitative illustration in the report shows that, for an MLP with structure `INPUT → 1024 → 512 → 256 → 3`, moving from sparse TF-IDF input to a 768-d BERT embedding reduces parameters by ~**86.75%**, improving training stability and lowering compute cost.

---

## Q6–Q7 — Transformer Fine-Tuning with LoRA (Best Model)

Instead of relying on fixed embeddings with a shallow head, we fine-tune a pretrained transformer for **sequence classification** using **LoRA (Low-Rank Adaptation)**:

- Backbone weights are frozen.
- Trainable low-rank matrices are injected into key attention projections.
- This yields strong task adaptation with substantially fewer trainable parameters than full fine-tuning.

### LoRA Configuration

The LoRA setup is designed to balance adaptation capacity and generalization:

- `task_type = SEQ_CLS`
- `target_modules = ["query", "value"]`
- `r = 8`
- `lora_alpha = 16`
- `lora_dropout = 0.1`

### Results and Error Profile

The fine-tuned **BERT + LoRA (SEQ_CLS)** model achieves:

- **Accuracy = 0.761**
- **Macro-Recall = 0.760**
- **Macro-F1 = 0.763**
- Weighted **ROC-AUC = 0.901** (strong global separability)

Compared to the prior BERT embedding baseline `LinearHead(768→3)` (Acc 0.664, Macro-F1 0.666), LoRA improves Macro-F1 by **+0.097** (≈ +14.6% relative). The remaining confusions concentrate more heavily around the **neutral class**, which behaves as an “intermediate / ambiguous” region between polarized sentiments.

---

## Results Overview

### Best models (test set)

| Category | Best configuration | Accuracy | Macro-Recall | Macro-F1 |
|---|---:|---:|---:|---:|
| **Best overall** | **BERT + LoRA (SEQ_CLS)** | **0.761** | **0.760** | **0.763** |
| Best classical | TF-IDF (char) + Random Forest | 0.720 | 0.719 | 0.722 |
| Best MLP (sparse) | Char TF-IDF + MLP (4096–2048–1024, drop 0.10, ReLU) | 0.708 | 0.706 | 0.711 |
| Best BERT “head-only” baseline | BERT + LinearHead (768→3) | 0.664 | 0.662 | 0.666 |
| Best LLM (few-shot) | gemma-3-4b-it (API) | 0.632 | — | 0.630 |

### What the ranking tells us

- **Character TF-IDF dominates** in classical and MLP settings for tweets, likely due to robustness to informal spelling and subword cues.
- **MLPs can match strong classical baselines**, but gains are not guaranteed: sparse high-dimensional inputs can overfit easily.
- **Few-shot LLM inference** is competitive but underperforms specialized fine-tuned encoders for this dataset.
- **LoRA fine-tuning is the turning point:** adapting the encoder (even lightly) beats all classical and MLP baselines.

---

## Repository Usage

### Environment

Typical stack for reproducing the experiments:

- Python 3.x
- Core: `numpy`, `pandas`, `scikit-learn`
- NLP preprocessing: `nltk`
- Plotting: `matplotlib`, `seaborn`
- Neural models: `torch`
- Transformers: `transformers`, `peft` (LoRA), `accelerate` (recommended)

> If your repository includes `requirements.txt` / `environment.yml`, prefer that as the single source of truth.

### Artifacts and Output Conventions

Common artifacts produced by the pipeline:

- **GridSearch CSV logs**: full candidate tables with hyperparameters and timing fields.
- **Saved classical models**: `joblib` pipelines.
- **Saved MLP checkpoints**: `.pt` files under `outputs/mlp/`.
- **Figures**: confusion matrices, metric comparisons, ROC curves.

If your repo follows the convention used in the evaluation scripts, classical “best” models may be stored like:

- `outputs/best__{vectorizer}__{model}.joblib`  
  Example vectorizers: `bow`, `tfidf_word`, `tfidf_char`

### Reproducibility Checklist

1. **Prepare data**
   - Place raw dataset locally (e.g., `data/raw/`).
   - Run preprocessing to produce the processed splits (e.g., `*_df_processed.csv`).

2. **Reproduce classical baselines**
   - Run GridSearch for each pipeline family.
   - Export the full CV results to CSV.
   - Save the best estimator to `outputs/`.

3. **Reproduce MLP experiments**
   - Train the selected architectures per representation family.
   - Save `.pt` checkpoints and metric summaries.

4. **Reproduce BERT / LoRA**
   - Run the notebook/script that performs tokenization, Trainer setup, and evaluation.
   - Save the final LoRA-adapted model (and tokenizer) to `outputs/`.

---

## Key Takeaways

- For tweet sentiment, **Char TF-IDF + strong classical classifiers** provides an unusually high baseline and should be treated as a serious reference point.
- MLPs on sparse inputs can **memorize extremely well**, so test performance must be interpreted alongside generalization controls.
- Few-shot LLMs are attractive for “zero-training” deployments, but **specialization wins** here.
- **BERT + LoRA** achieves the best overall performance and is the recommended model for this project.

---

## Authors

- Carlos Adrian Meneses Gamboa
- Jose Daniel Chacon Gomez
- Santiago Florido Gomez