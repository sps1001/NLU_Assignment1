# CSL7640 — Assignment 1, Problem 4
# Sports vs Politics Document Classifier

**Student:** Sahilpreet Singh (B23CS1061)  
**Course:** CSL7640 — Natural Language Understanding, IIT Jodhpur  
**Date:** February 2026

---

## 1. Introduction

This report presents a text document classification system that categorizes documents into two classes: **Sports** and **Politics**. The task is to design, implement, and compare at least three machine learning techniques using different feature representations including n-grams, TF-IDF, and Bag of Words. All implementations are done from scratch using only Python's standard library to demonstrate a deep understanding of the underlying algorithms.

The motivation behind this task is to understand how classical machine learning techniques perform on a fundamental NLP problem — document classification. By comparing multiple approaches, we can gain insights into the strengths and weaknesses of different feature representations and classification algorithms.

---

## 2. Data Collection and Dataset Description

### 2.1 Data Collection Method

The dataset was manually curated to ensure high quality and diversity. Each sentence was written to represent a typical news headline or summary that would belong to either the Sports or Politics category. Manual curation was chosen over web scraping to avoid copyright issues and to ensure clean, well-formed sentences without noise.

### 2.2 Dataset Statistics

| Property | Value |
|---|---|
| Total samples | 100 |
| Sports samples | 50 |
| Politics samples | 50 |
| Class balance | Perfectly balanced (50:50) |
| Average sentence length | ~10-12 words |
| Train-test split | 80:20 (80 train, 20 test) |

### 2.3 Dataset Analysis

**Sports sub-topics covered:**
- Cricket (IPL, Test matches, ICC events)
- Football (Premier League, Champions League, FIFA)
- Tennis (Grand Slams, ATP/WTA)
- Basketball (NBA)
- Olympics (multiple sports)
- Formula 1, Boxing, Badminton, Hockey, Cycling

**Politics sub-topics covered:**
- Elections (state, national, local body)
- Parliamentary proceedings
- Policy and legislation
- International diplomacy
- Governance and administration
- Defense and security
- Economic policy and budget

The diversity in sub-topics ensures that the classifier learns generalizable patterns rather than memorizing specific keywords. The dataset includes both Indian and international contexts for both categories.

### 2.4 Vocabulary Analysis

After preprocessing (lowercasing, removing punctuation), the dataset yields:
- **Unigram vocabulary size:** 388 unique words
- **Unigram + Bigram vocabulary size:** 964 unique tokens

Key discriminative words include sports-specific terms (e.g., "won", "match", "championship", "team", "medal") and politics-specific terms (e.g., "government", "minister", "parliament", "election", "policy").

---

## 3. Feature Representations

Three different feature extraction methods were implemented:

### 3.1 Bag of Words (BoW)

Bag of Words is the simplest feature representation. Each document is represented as a vector of word counts, where each dimension corresponds to a word in the vocabulary.

**Formula:**  
For a document d and vocabulary word w:  
`BoW(d, w) = count of w in d`

**Advantages:**
- Simple to implement and understand
- Works well for small datasets
- No hyperparameters for feature extraction

**Disadvantages:**
- Ignores word order
- Does not account for document length differences
- Common words may dominate the representation

### 3.2 TF-IDF (Term Frequency — Inverse Document Frequency)

TF-IDF weights each word by how important it is to a specific document relative to the entire corpus.

**Formula:**  
`TF(w, d) = count(w in d) / total_words(d)`  
`IDF(w) = log((N + 1) / (df(w) + 1)) + 1`  
`TF-IDF(w, d) = TF(w, d) × IDF(w)`

Where N is total documents and df(w) is the number of documents containing w.

**Advantages:**
- Down-weights common words like "the", "is"
- Normalizes for document length
- Highlights discriminative words

**Disadvantages:**
- Still ignores word order
- IDF can be sensitive to small corpora

### 3.3 N-gram Features (Unigram + Bigram)

N-gram features capture sequences of consecutive words, adding context that single words alone cannot provide.

**Example:**  
Sentence: "India won the match"  
Unigrams: ["india", "won", "the", "match"]  
Bigrams: ["india won", "won the", "the match"]  
Combined: all of the above

**Advantages:**
- Captures word-pair context (e.g., "prime minister" vs "prime" alone)
- Can distinguish phrases that have different meanings

**Disadvantages:**
- Much larger feature space (964 vs 388 for unigrams only)
- Increased sparsity, which can hurt some classifiers
- Higher computation cost

---

## 4. Machine Learning Techniques

### 4.1 Multinomial Naive Bayes

Naive Bayes is a probabilistic classifier based on Bayes' theorem with the "naive" assumption that features are conditionally independent given the class.

**Classification Rule:**  
`P(class | document) ∝ P(class) × ∏ P(word | class)`

**Laplace Smoothing:**  
`P(w | c) = (count(w, c) + 1) / (total_words(c) + |V|)`

This prevents zero probabilities for unseen words. Log probabilities are used to avoid floating-point underflow.

**Implementation Details:**
- Prior probabilities computed from training class frequencies
- Feature probabilities stored per class per vocabulary word
- Prediction uses log-sum of priors and feature likelihoods

### 4.2 Logistic Regression

Logistic Regression is a discriminative linear classifier that models the probability of a class using the sigmoid function.

**Model:**  
`P(y=1 | x) = σ(w·x + b) = 1 / (1 + exp(-(w·x + b)))`

**Training:** Gradient descent on binary cross-entropy loss:
- For each sample, compute prediction error
- Update weights: `w = w - lr × error × x`
- Update bias: `b = b - lr × error`

**Implementation Details:**
- Learning rate: 0.1
- Training epochs: 300
- Weights initialized randomly (seed=42)
- Sigmoid clamped to [-500, 500] to prevent overflow

### 4.3 K-Nearest Neighbors (KNN)

KNN is a non-parametric, lazy learning algorithm that classifies a document based on the majority class among its K nearest neighbors.

**Distance Metric — Cosine Similarity:**  
`sim(a, b) = (a · b) / (||a|| × ||b||)`

Cosine similarity is preferred over Euclidean distance for text because it measures orientation rather than magnitude, making it robust to varying document lengths.

**Implementation Details:**
- K = 5 (odd number to avoid ties)
- Lazy learning: no training phase, stores all training data
- Prediction: compute similarity to all training samples, majority vote among top-K

---

## 5. Results and Quantitative Comparison

### 5.1 Individual Results

#### Experiment 1: Naive Bayes + Bag of Words

| Metric | SPORTS | POLITICS |
|---|---|---|
| Precision | 1.0000 | 0.8182 |
| Recall | 0.8182 | 1.0000 |
| F1-Score | 0.9000 | 0.9000 |
| **Accuracy** | **90.00%** | |

**Confusion Matrix:**

| | Pred: SPORTS | Pred: POLITICS |
|---|---|---|
| True: SPORTS | 9 | 2 |
| True: POLITICS | 0 | 9 |

#### Experiment 2: Logistic Regression + TF-IDF

| Metric | SPORTS | POLITICS |
|---|---|---|
| Precision | 1.0000 | 0.8182 |
| Recall | 0.8182 | 1.0000 |
| F1-Score | 0.9000 | 0.9000 |
| **Accuracy** | **90.00%** | |

**Confusion Matrix:**

| | Pred: SPORTS | Pred: POLITICS |
|---|---|---|
| True: SPORTS | 9 | 2 |
| True: POLITICS | 0 | 9 |

#### Experiment 3: KNN (k=5) + Bigram Features

| Metric | SPORTS | POLITICS |
|---|---|---|
| Precision | 1.0000 | 0.6923 |
| Recall | 0.6364 | 1.0000 |
| F1-Score | 0.7778 | 0.8182 |
| **Accuracy** | **80.00%** | |

**Confusion Matrix:**

| | Pred: SPORTS | Pred: POLITICS |
|---|---|---|
| True: SPORTS | 7 | 4 |
| True: POLITICS | 0 | 9 |

### 5.2 Comparative Summary

| Technique | Feature | Accuracy | F1 (Sports) | F1 (Politics) |
|---|---|---|---|---|
| Naive Bayes | Bag of Words | **90.00%** | **0.9000** | **0.9000** |
| Logistic Regression | TF-IDF | **90.00%** | **0.9000** | **0.9000** |
| KNN (k=5) | Bigrams | 80.00% | 0.7778 | 0.8182 |

### 5.3 Analysis of Results

1. **Naive Bayes and Logistic Regression tied at 90%** — Both methods perform equally well on this dataset. NB benefits from the conditional independence assumption working reasonably well for text, while LR benefits from learning discriminative weights via TF-IDF features.

2. **KNN performed worst at 80%** — The larger bigram feature space (964 dimensions) creates sparse vectors, making cosine similarity less reliable. KNN is also sensitive to the choice of K and does not learn feature importance.

3. **Perfect precision for SPORTS across all methods** — When any classifier predicts SPORTS, it is always correct. However, some SPORTS documents are misclassified as POLITICS (lower recall). This suggests some sports sentences contain words that overlap with political vocabulary.

4. **Perfect recall for POLITICS** — All politics documents are correctly identified by all three classifiers, indicating strong discriminative vocabulary in the politics class.

---

## 6. Limitations

1. **Small Dataset Size:** With only 50 samples per class, the models have limited training data. Larger datasets would likely improve accuracy and generalizability.

2. **No Cross-Validation:** A single 80/20 split is used. K-fold cross-validation would provide more robust accuracy estimates and reduce variance in the results.

3. **Surface-Level Features Only:** All three techniques use word-level or n-gram statistics. They cannot capture semantic meaning, word embeddings, or contextual relationships between words.

4. **Binary Classification Only:** The system only handles two classes. Extending to multi-class classification (e.g., adding Technology, Entertainment, Finance) would require modifications.

5. **Ambiguous Texts:** Documents that mix domains (e.g., "The sports minister announced new funding for athletes") may be misclassified because they contain vocabulary from both categories.

6. **No Preprocessing Beyond Lowercasing:** Techniques like stemming, lemmatization, or stopword removal could potentially improve performance by reducing vocabulary size and noise.

7. **Manual Dataset:** The dataset was manually created and may not fully represent real-world news articles in terms of style, length, and complexity.

---

## 7. Conclusion

This project demonstrates that even simple, from-scratch implementations of classical ML techniques can achieve strong performance on text classification tasks. Naive Bayes with Bag of Words and Logistic Regression with TF-IDF both achieve 90% accuracy, while KNN with bigram features achieves 80%.

The key takeaway is that for well-separated classes like Sports and Politics, feature representation matters as much as the choice of classifier. BoW and TF-IDF both capture the discriminative vocabulary effectively, while bigrams, despite providing more context, introduce sparsity that hurts KNN performance.

For real-world applications, deep learning approaches (RNNs, Transformers) with word embeddings (Word2Vec, BERT) would significantly outperform these methods by capturing semantic meaning and long-range dependencies.

---

## 8. References

1. Manning, C.D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press.
2. Jurafsky, D., & Martin, J.H. (2023). *Speech and Language Processing.* 3rd Edition.
3. Sebastiani, F. (2002). Machine learning in automated text categorization. *ACM Computing Surveys*, 34(1), 1-47.
