# Sports vs Politics Document Classifier

**Roll Number:** B23CS1061  
**Course:** CSL7640 — Natural Language Understanding  
**Assignment:** 1, Problem 4

---

## Overview

A text document classifier that classifies input documents as **Sports** or **Politics** using three different machine learning techniques, all implemented from scratch using only Python's standard library.

## Dataset

- **Sports:** 50 manually curated sentences about cricket, football, tennis, Olympics, NBA, F1, etc.
- **Politics:** 50 manually curated sentences about elections, parliament, policies, diplomacy, governance, etc.
- **Split:** 80% training, 20% testing (fixed seed for reproducibility)

### Data Collection

The dataset was created manually to ensure diversity across sub-topics:

| Category | Sub-topics Covered |
|---|---|
| Sports | Cricket, Football, Tennis, Basketball, Olympics, F1, Boxing, Badminton, Hockey, Cycling |
| Politics | Elections, Parliament, Policy, Diplomacy, Governance, Protests, Defense, Trade, Judiciary |

## Techniques Compared

### 1. Naive Bayes + Bag of Words (BoW)
- **Feature Representation:** Binary/count bag-of-words vectors
- **Classifier:** Multinomial Naive Bayes with Laplace smoothing
- **Strengths:** Fast training, works well with small datasets, probabilistic interpretation
- **Limitations:** Assumes feature independence (naive assumption)

### 2. Logistic Regression + TF-IDF
- **Feature Representation:** TF-IDF (Term Frequency × Inverse Document Frequency)
- **Classifier:** Binary logistic regression with sigmoid + gradient descent
- **Strengths:** Learns feature importance, handles varying document lengths via TF-IDF normalization
- **Limitations:** Requires careful learning rate tuning, may converge slowly

### 3. K-Nearest Neighbors (KNN) + Bigram Features
- **Feature Representation:** Unigram + Bigram count vectors
- **Classifier:** KNN (k=5) using cosine similarity
- **Strengths:** Non-parametric, captures word-pair context via bigrams
- **Limitations:** Slow at prediction time, sensitive to k value and feature sparsity

## How to Run

```bash
python b23cs1061_prob4.py
```

Make sure `sports.txt` and `politics.txt` are in the same directory.

## Output

The script prints:
1. Dataset statistics
2. Per-experiment results: accuracy, precision, recall, F1-score, confusion matrix
3. Final comparison table of all three techniques

## Project Structure

```
exp4/
├── b23cs1061_prob4.py    # Main classifier script
├── sports.txt            # Sports training data (50 sentences)
├── politics.txt          # Politics training data (50 sentences)
├── README.md             # This file (GitHub page)
└── report.md             # Detailed report (for PDF conversion)
```

## Results Summary

| Technique | Feature | Accuracy |
|---|---|---|
| Naive Bayes | Bag of Words | **90.00%** |
| Logistic Regression | TF-IDF | **90.00%** |
| KNN (k=5) | Bigrams | 80.00% |

## Limitations

1. **Small dataset** — Only 50 samples per class; performance would improve with more data
2. **No deep features** — All techniques use surface-level word statistics
3. **No cross-validation** — Uses a single 80/20 split; k-fold would give more robust estimates
4. **Domain-specific vocabulary** — Classifier may struggle with ambiguous sentences (e.g., "The minister played cricket")
5. **No word embeddings** — Techniques don't capture semantic similarity between words

## Author

Sahilpreet Singh (B23CS1061)  
IIT Jodhpur — CSL7640 Natural Language Understanding
