# Sports vs Politics Document Classifier

**Roll Number:** B23CS1061  
**Course:** CSL7640 — Natural Language Understanding  
**Assignment:** 1, Problem 4

---

## Overview

A text document classifier that classifies input documents as **Sports** or **Politics** using three different machine learning techniques, all implemented from scratch using only Python's standard library.

## Dataset

The dataset is sourced from the **[News Category Dataset on Kaggle](https://www.kaggle.com/datasets/setseries/news-category-dataset)**, originally from HuffPost (2012-2018). The CSV file (`newscategorizer.csv`) contains 50,000 news articles across 10 balanced categories (5000 each).

Our program reads the CSV directly, filters **SPORTS** and **POLITICS** categories, and uses the `short_description` field as text for classification.

### Dataset Details

| Property | Value |
|---|---|
| Source | Kaggle — News Category Dataset |
| Original Publisher | HuffPost |
| License | CC0 — Public Domain |
| Total rows in CSV | 50,000 (10 categories × 5000) |
| Categories used | SPORTS, POLITICS |
| Samples per class | 200 |
| Total samples used | 400 |
| Text field | `short_description` (fallback: `headline`) |
| Train-test split | 80:20 (320 train, 80 test) |

### All Categories in Original Dataset

| Category | Samples |
|---|---|
| Business | 5000 |
| Entertainment | 5000 |
| Food & Drink | 5000 |
| Parenting | 5000 |
| **Politics** | **5000** |
| **Sports** | **5000** |
| Style & Beauty | 5000 |
| Travel | 5000 |
| Wellness | 5000 |
| World News | 5000 |

## Techniques Compared

### 1. Naive Bayes + Bag of Words (BoW)
- **Feature:** Count-based bag-of-words vectors (2269 unigrams)
- **Classifier:** Multinomial Naive Bayes with Laplace smoothing
- **Strengths:** Fast training, handles sparse features well, probabilistic interpretation
- **Limitations:** Assumes feature independence (naive assumption)

### 2. Logistic Regression + TF-IDF
- **Feature:** TF-IDF weighted vectors (Term Frequency × Inverse Document Frequency)
- **Classifier:** Binary logistic regression with sigmoid + gradient descent
- **Strengths:** Learns feature importance, TF-IDF normalizes for document length
- **Limitations:** Requires learning rate tuning, may need more epochs on large vocabularies

### 3. K-Nearest Neighbors (KNN) + Bigram Features
- **Feature:** Unigram + Bigram count vectors (6881 features)
- **Classifier:** KNN (k=5) using cosine similarity
- **Strengths:** Non-parametric, captures word-pair context via bigrams
- **Limitations:** Slow at prediction time, sparse bigram features reduce similarity effectiveness

## How to Run

```bash
python b23cs1061_prob4.py
```

Make sure `newscategorizer.csv` is in the same directory as the script.

## Output

The script prints:
1. Dataset statistics (samples loaded, train/test split)
2. Per-experiment results: accuracy, precision, recall, F1-score, confusion matrix
3. Final comparison table of all three techniques

## Project Structure

```
exp4/
├── b23cs1061_prob4.py        # Main classifier script (reads CSV directly)
├── newscategorizer.csv       # Kaggle News Category Dataset
├── README.md                 # This file (GitHub page)
└── report.md                 # Detailed report (for PDF conversion)
```

## Results Summary

| Technique | Feature | Accuracy | F1 (Sports) | F1 (Politics) |
|---|---|---|---|---|
| **Naive Bayes** | Bag of Words | **80.00%** | 0.7647 | 0.8261 |
| Logistic Regression | TF-IDF | 77.50% | 0.6250 | 0.7500 |
| KNN (k=5) | Bigrams | 65.00% | 0.6216 | 0.6744 |

### Confusion Matrices

**Naive Bayes + BoW (Best: 80%)**

| | Pred: SPORTS | Pred: POLITICS |
|---|---|---|
| True: SPORTS | 26 | 9 |
| True: POLITICS | 7 | 38 |

**Logistic Regression + TF-IDF (77.50%)**

| | Pred: SPORTS | Pred: POLITICS |
|---|---|---|
| True: SPORTS | 10 | 9 |
| True: POLITICS | 3 | 18 |

**KNN + Bigrams (65%)**

| | Pred: SPORTS | Pred: POLITICS |
|---|---|---|
| True: SPORTS | 23 | 12 |
| True: POLITICS | 16 | 29 |

## Limitations

1. **Standard library only** — No NumPy/scikit-learn; production systems would use optimized libraries
2. **Limited preprocessing** — Only lowercasing and punctuation removal; no stemming, lemmatization, or stopword removal
3. **No cross-validation** — Single 80/20 split; k-fold would give more robust accuracy estimates
4. **Binary classification** — Only 2 of 10 available categories used; extending to multi-class would need modifications
5. **No word embeddings** — Surface-level word statistics cannot capture semantic similarity
6. **Real-world noise** — Some articles mix sports and politics (e.g., "sports policy"), causing misclassification

## Author

Sahilpreet Singh (B23CS1061)  
IIT Jodhpur — CSL7640 Natural Language Understanding
