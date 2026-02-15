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

The dataset used for this project is the **News Category Dataset** obtained from Kaggle. This is a publicly available dataset originally sourced from HuffPost (HuffPost News Category Dataset). The dataset was uploaded by the user **CODIFY** on Kaggle and contains news headlines and short descriptions from the year 2012 to 2018.

**Dataset Source:** [https://www.kaggle.com/datasets/setseries/news-category-dataset](https://www.kaggle.com/datasets/setseries/news-category-dataset)  
**License:** CC0 — Public Domain  
**Original Publisher:** HuffPost

The original dataset is a follow-up to the News Category Dataset. It contains approximately 45,500 news headlines with 5 columns. The motive of this dataset was to give beginners an easy-to-use dataset. The dataset has been cleaned, filtered and the target feature has been balanced, unlike the original dataset.

### 2.2 Original Dataset Overview

The full CSV file (`newscategorizer.csv`) contains the following structure:

| Column | Description |
|---|---|
| `category` | News category label (10 categories) |
| `headline` | Short headline of the article |
| `links` | URL link to the original article |
| `short_description` | Longer description/summary of the article |
| `keywords` | Main keywords extracted from the URL |

The dataset has **10 balanced categories**, each with 5000 samples:

| Category | Number of Samples |
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
| **Total** | **50,000** |

### 2.3 Data Extraction and Preprocessing

Since our task is binary classification (Sports vs Politics), I wrote code to filter only the **SPORTS** and **POLITICS** rows from the CSV. The program reads the CSV file using Python's built-in `csv` module and extracts 200 samples per category (400 total).

I chose to use the `short_description` field as the text for classification because headlines are too short (3-8 words) and don't provide enough discriminative context. When `short_description` is empty, the code falls back to using the `headline`.

**Preprocessing steps applied:**
1. Read CSV using `csv.DictReader`
2. Filter rows where `category` is "SPORTS" or "POLITICS"
3. Use `short_description` as text (fallback to `headline`)
4. Clean up extra whitespace and newlines
5. Take 200 samples per class for balanced representation
6. Lowercase all text and remove punctuation during feature extraction

### 2.4 Final Dataset Statistics

| Property | Value |
|---|---|
| Total samples used | 400 |
| Sports samples | 200 |
| Politics samples | 200 |
| Class balance | Perfectly balanced (50:50) |
| Source | Kaggle - News Category Dataset |
| Text field used | `short_description` |
| Train-test split | 80:20 (320 train, 80 test) |
| Random seed | 42 (for reproducibility) |

### 2.5 Dataset Analysis

**Sports articles cover topics such as:**
- NFL, NBA, MLB, NHL game results and player performances
- FIFA World Cup and international football
- Olympics coverage and medal tallies
- Boxing, tennis, golf, and motorsport events
- Player trades, injuries, and team management

**Politics articles cover topics such as:**
- U.S. congressional proceedings and legislation
- Presidential campaigns and election coverage
- White House policies and executive actions
- International diplomacy and foreign affairs
- State and local government issues
- Supreme Court decisions and judicial matters

### 2.6 Vocabulary Analysis

After preprocessing (lowercasing, removing punctuation), the dataset yields:

| Feature Type | Vocabulary Size |
|---|---|
| Unigrams only | 2269 |
| Unigrams + Bigrams | 6881 |

The vocabulary is significantly larger than a hand-crafted dataset because real news articles use diverse phrasing, proper nouns, and domain-specific terminology. Key discriminative words include sports terms (e.g., "game", "season", "team", "championship", "scored") and politics terms (e.g., "trump", "president", "senate", "legislation", "campaign").

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
- Works well as a baseline
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
- Much larger feature space (6881 vs 2269 for unigrams only)
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
| Precision | 0.7879 | 0.8085 |
| Recall | 0.7429 | 0.8444 |
| F1-Score | 0.7647 | 0.8261 |
| **Accuracy** | **80.00%** | |

**Confusion Matrix:**

| | Pred: SPORTS | Pred: POLITICS |
|---|---|---|
| True: SPORTS | 26 | 9 |
| True: POLITICS | 7 | 38 |

#### Experiment 2: Logistic Regression + TF-IDF

| Metric | SPORTS | POLITICS |
|---|---|---|
| Precision | 0.7692 | 0.6667 |
| Recall | 0.5263 | 0.8571 |
| F1-Score | 0.6250 | 0.7500 |
| **Accuracy** | **77.50%** | |

**Confusion Matrix:**

| | Pred: SPORTS | Pred: POLITICS |
|---|---|---|
| True: SPORTS | 10 | 9 |
| True: POLITICS | 3 | 18 |

#### Experiment 3: KNN (k=5) + Bigram Features

| Metric | SPORTS | POLITICS |
|---|---|---|
| Precision | 0.5897 | 0.7073 |
| Recall | 0.6571 | 0.6444 |
| F1-Score | 0.6216 | 0.6744 |
| **Accuracy** | **65.00%** | |

**Confusion Matrix:**

| | Pred: SPORTS | Pred: POLITICS |
|---|---|---|
| True: SPORTS | 23 | 12 |
| True: POLITICS | 16 | 29 |

### 5.2 Comparative Summary

| Technique | Feature | Accuracy | F1 (Sports) | F1 (Politics) |
|---|---|---|---|---|
| **Naive Bayes** | Bag of Words | **80.00%** | **0.7647** | **0.8261** |
| Logistic Regression | TF-IDF | 77.50% | 0.6250 | 0.7500 |
| KNN (k=5) | Bigrams | 65.00% | 0.6216 | 0.6744 |

### 5.3 Analysis of Results

1. **Naive Bayes performed best at 80%** — Naive Bayes with Bag of Words features gives the best accuracy on this real-world dataset. The conditional independence assumption works reasonably well for text classification because the presence/absence of discriminative words independently contributes to the class prediction.

2. **Logistic Regression achieved 77.50%** — LR with TF-IDF features performs slightly lower than NB. With real-world data, TF-IDF normalization helps reduce the impact of common words, but the gradient descent optimization with 300 epochs may not fully converge on the larger vocabulary (2269 features).

3. **KNN performed worst at 65%** — The bigram feature space (6881 dimensions) creates extremely sparse vectors, making cosine similarity less effective. KNN is also a lazy learner that doesn't learn feature importance — it treats all dimensions equally, which hurts performance when most features are zero.

4. **Politics has better recall across methods** — This suggests politics articles have more distinctive vocabulary (terms like "trump", "senate", "legislation") that classifiers can reliably pick up. Sports articles sometimes use more general language that overlaps with other domains.

5. **Real-world data is harder** — Compared to manually curated datasets, real HuffPost articles contain more noise, ambiguous phrasing, and overlapping vocabulary. Some articles cover topics like "sports politics" or political commentary on sports events, making clean classification harder.

---

## 6. Limitations

1. **Standard Library Only:** All implementations use only Python's standard library. Production systems would use optimized libraries like scikit-learn, which provide better numerical stability and performance.

2. **Limited Preprocessing:** Only lowercasing and punctuation removal are applied. Techniques like stemming, lemmatization, stopword removal, or named entity recognition could improve accuracy by reducing noise.

3. **No Cross-Validation:** A single 80/20 split is used. K-fold cross-validation would provide more robust accuracy estimates and reduce variance in the results.

4. **Binary Classification Only:** The system only handles two classes (Sports and Politics). The original dataset has 10 categories — extending to multi-class would require modifications to the logistic regression (one-vs-rest) and evaluation metrics.

5. **Ambiguous Articles:** Some real-world news articles mix domains. For example, articles about "sports policy", "athlete activism", or "government sports funding" contain vocabulary from both categories, leading to misclassification.

6. **No Word Embeddings:** All techniques use surface-level word statistics. They cannot capture semantic similarity — for example, "baseball" and "cricket" are semantically similar (both sports) but treated as completely independent features.

7. **Sample Size Trade-off:** Using 200 samples per class is a balance between having enough data and keeping computation reasonable. Using all 5000 per class would improve accuracy but significantly slow down KNN predictions.

8. **Fixed Hyperparameters:** Learning rate (0.1), epochs (300), and K (5) were chosen based on common defaults. Hyperparameter tuning could potentially improve individual model performance.

---

## 7. Conclusion

This project demonstrates that from-scratch implementations of classical ML techniques can achieve reasonable performance on real-world text classification tasks. Naive Bayes with Bag of Words achieves the best accuracy at 80%, followed by Logistic Regression with TF-IDF at 77.50%, and KNN with bigram features at 65%.

The key takeaway is that simpler models (Naive Bayes) can outperform more complex ones (KNN with bigrams) when the feature space is large and sparse. Bag of Words, despite being the simplest representation, proves to be the most effective when combined with a probabilistic classifier that can handle feature sparsity gracefully.

Using a real-world dataset from Kaggle (HuffPost News Category Dataset) provided more realistic and challenging evaluation compared to manually curated sentences. The results highlight that real-world text classification involves dealing with noisy, ambiguous, and overlapping vocabularies that surface-level features alone cannot fully resolve.

For production applications, deep learning approaches (RNNs, Transformers) with pre-trained word embeddings (Word2Vec, GloVe, BERT) would significantly outperform these methods by capturing semantic meaning and contextual relationships.

---

## 8. References

1. Manning, C.D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press.
2. Jurafsky, D., & Martin, J.H. (2023). *Speech and Language Processing.* 3rd Edition.
3. Sebastiani, F. (2002). Machine learning in automated text categorization. *ACM Computing Surveys*, 34(1), 1-47.
4. News Category Dataset — Kaggle. https://www.kaggle.com/datasets/setseries/news-category-dataset
5. HuffPost — Original source of the news articles (2012-2018).
