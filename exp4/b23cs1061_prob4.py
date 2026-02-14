# b23cs1061_prob4.py
# Sports vs Politics Document Classifier
# Roll Number: B23CS1061
# Course: CSL7640 - Natural Language Understanding
#
# This script compares three ML techniques for classifying text documents
# as either Sports or Politics using different feature representations.
#
# Usage: python b23cs1061_prob4.py

import os
import re
import math
import random
from collections import Counter, defaultdict


# ========================================
# DATA LOADING AND PREPROCESSING
# ========================================

def load_dataset(sports_file, politics_file):
    """Load sports and politics text files, return list of (text, label) tuples."""
    data = []

    with open(sports_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append((line, "SPORTS"))

    with open(politics_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append((line, "POLITICS"))

    return data


def preprocess(text):
    """Lowercase and remove punctuation, return list of words."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    words = text.split()
    return words


def get_ngrams(words, n):
    """Generate n-grams from a list of words."""
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i:i + n])
        ngrams.append(ngram)
    return ngrams


def train_test_split(data, train_ratio=0.8, seed=42):
    """Shuffle and split data into train and test."""
    shuffled = data[:]
    random.seed(seed)
    random.shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    return shuffled[:split], shuffled[split:]


# ========================================
# FEATURE EXTRACTION
# ========================================

def build_vocabulary(train_data, use_ngrams=1):
    """Build vocabulary from training data. use_ngrams=1 for unigrams, 2 for bigrams etc."""
    vocab = set()
    for text, label in train_data:
        words = preprocess(text)
        if use_ngrams == 1:
            for w in words:
                vocab.add(w)
        else:
            for n in range(1, use_ngrams + 1):
                for ng in get_ngrams(words, n):
                    vocab.add(ng)
    return sorted(vocab)


def bag_of_words(text, vocab):
    """Convert text to bag-of-words vector using given vocabulary."""
    words = preprocess(text)
    word_counts = Counter(words)
    vector = []
    for v in vocab:
        vector.append(word_counts.get(v, 0))
    return vector


def compute_tf(text, vocab):
    """Compute term frequency for each vocab word in the text."""
    words = preprocess(text)
    word_counts = Counter(words)
    total = len(words) if len(words) > 0 else 1
    tf = []
    for v in vocab:
        tf.append(word_counts.get(v, 0) / total)
    return tf


def compute_idf(train_data, vocab):
    """Compute inverse document frequency for each vocab word."""
    N = len(train_data)
    doc_counts = defaultdict(int)

    for text, label in train_data:
        words_in_doc = set(preprocess(text))
        for v in vocab:
            if v in words_in_doc:
                doc_counts[v] += 1

    idf = []
    for v in vocab:
        # add 1 to avoid division by zero (smoothing)
        idf_val = math.log((N + 1) / (doc_counts[v] + 1)) + 1
        idf.append(idf_val)
    return idf


def compute_tfidf(text, vocab, idf_values):
    """Compute TF-IDF vector for a document."""
    tf = compute_tf(text, vocab)
    tfidf = []
    for i in range(len(vocab)):
        tfidf.append(tf[i] * idf_values[i])
    return tfidf


def ngram_features(text, vocab, n=2):
    """Convert text to n-gram based feature vector."""
    words = preprocess(text)
    all_grams = []
    for i in range(1, n + 1):
        all_grams.extend(get_ngrams(words, i))
    gram_counts = Counter(all_grams)
    vector = []
    for v in vocab:
        vector.append(gram_counts.get(v, 0))
    return vector


# ========================================
# CLASSIFIER 1: NAIVE BAYES
# ========================================

class NaiveBayesClassifier:
    """Multinomial Naive Bayes with Laplace smoothing."""

    def __init__(self):
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = []

    def train(self, X, y):
        self.classes = list(set(y))
        n_samples = len(y)

        for c in self.classes:
            # get indices of samples belonging to this class
            class_indices = [i for i in range(n_samples) if y[i] == c]
            self.class_priors[c] = len(class_indices) / n_samples

            # sum up feature counts for this class
            n_features = len(X[0])
            feature_sums = [0] * n_features
            for idx in class_indices:
                for j in range(n_features):
                    feature_sums[j] += X[idx][j]

            total = sum(feature_sums)
            # Laplace smoothed probabilities
            self.feature_probs[c] = []
            for j in range(n_features):
                prob = (feature_sums[j] + 1) / (total + n_features)
                self.feature_probs[c].append(prob)

    def predict(self, X):
        predictions = []
        for sample in X:
            best_class = None
            best_score = float("-inf")
            for c in self.classes:
                score = math.log(self.class_priors[c])
                for j in range(len(sample)):
                    if sample[j] > 0:
                        score += sample[j] * math.log(self.feature_probs[c][j])
                if score > best_score:
                    best_score = score
                    best_class = c
            predictions.append(best_class)
        return predictions


# ========================================
# CLASSIFIER 2: LOGISTIC REGRESSION (from scratch)
# ========================================

class LogisticRegressionClassifier:
    """Simple binary logistic regression using gradient descent."""

    def __init__(self, lr=0.01, epochs=200):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0
        self.label_map = {}
        self.inv_label_map = {}

    def sigmoid(self, z):
        # clip to prevent overflow
        z = max(-500, min(500, z))
        return 1.0 / (1.0 + math.exp(-z))

    def train(self, X, y):
        # map labels to 0 and 1
        unique_labels = list(set(y))
        self.label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
        self.inv_label_map = {0: unique_labels[0], 1: unique_labels[1]}
        y_binary = [self.label_map[label] for label in y]

        n_features = len(X[0])
        n_samples = len(X)

        # initialize weights to small random values
        random.seed(42)
        self.weights = [random.uniform(-0.01, 0.01) for _ in range(n_features)]
        self.bias = 0

        # gradient descent
        for epoch in range(self.epochs):
            for i in range(n_samples):
                # compute prediction
                z = self.bias
                for j in range(n_features):
                    z += self.weights[j] * X[i][j]
                pred = self.sigmoid(z)

                # compute error
                error = pred - y_binary[i]

                # update weights
                for j in range(n_features):
                    self.weights[j] -= self.lr * error * X[i][j]
                self.bias -= self.lr * error

    def predict(self, X):
        predictions = []
        for sample in X:
            z = self.bias
            for j in range(len(sample)):
                z += self.weights[j] * sample[j]
            pred = self.sigmoid(z)
            label = 1 if pred >= 0.5 else 0
            predictions.append(self.inv_label_map[label])
        return predictions


# ========================================
# CLASSIFIER 3: KNN (K-Nearest Neighbors)
# ========================================

class KNNClassifier:
    """K-Nearest Neighbors classifier using cosine similarity."""

    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def cosine_similarity(self, a, b):
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0
        return dot / (norm_a * norm_b)

    def train(self, X, y):
        # KNN is lazy — just store the training data
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for sample in X:
            # find distances to all training samples
            similarities = []
            for i in range(len(self.X_train)):
                sim = self.cosine_similarity(sample, self.X_train[i])
                similarities.append((sim, self.y_train[i]))

            # sort by similarity (highest first) and pick top K
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_k = similarities[:self.k]

            # majority vote
            votes = Counter([label for _, label in top_k])
            predictions.append(votes.most_common(1)[0][0])
        return predictions


# ========================================
# EVALUATION METRICS
# ========================================

def compute_accuracy(y_true, y_pred):
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true) if y_true else 0


def compute_precision_recall_f1(y_true, y_pred, positive_class):
    """Compute precision, recall, F1 for a given class."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p == positive_class)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_class and p == positive_class)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive_class and p != positive_class)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def confusion_matrix(y_true, y_pred, classes):
    """Build and return confusion matrix as dict."""
    matrix = {c1: {c2: 0 for c2 in classes} for c1 in classes}
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


def print_confusion_matrix(matrix, classes):
    """Pretty print the confusion matrix."""
    # header
    header = "Actual \\ Predicted"
    print(f"  {header:>20s}", end="")
    for c in classes:
        print(f"  {c:>10s}", end="")
    print()
    # rows
    for c1 in classes:
        print(f"  {c1:>20s}", end="")
        for c2 in classes:
            print(f"  {matrix[c1][c2]:>10d}", end="")
        print()


# ========================================
# MAIN - RUN ALL EXPERIMENTS
# ========================================

def run_experiment(classifier, clf_name, X_train, y_train, X_test, y_test, classes):
    """Train a classifier and evaluate it, returning results dict."""
    classifier.train(X_train, y_train)
    y_pred = classifier.predict(X_test)

    acc = compute_accuracy(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, classes)

    print(f"\n{'=' * 55}")
    print(f"  {clf_name}")
    print(f"{'=' * 55}")
    print(f"  Accuracy: {acc * 100:.2f}%")

    for c in classes:
        p, r, f1 = compute_precision_recall_f1(y_test, y_pred, c)
        print(f"  {c}: Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}")

    print(f"\n  Confusion Matrix:")
    print_confusion_matrix(cm, classes)

    return {"name": clf_name, "accuracy": acc, "predictions": y_pred}


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sports_file = os.path.join(script_dir, "sports.txt")
    politics_file = os.path.join(script_dir, "politics.txt")

    # load data
    print("Loading dataset...")
    data = load_dataset(sports_file, politics_file)
    print(f"  Total samples: {len(data)}")

    sports_count = sum(1 for _, l in data if l == "SPORTS")
    politics_count = sum(1 for _, l in data if l == "POLITICS")
    print(f"  Sports: {sports_count}, Politics: {politics_count}")

    # split into train and test
    train_data, test_data = train_test_split(data, train_ratio=0.8)
    print(f"\n  Train: {len(train_data)}, Test: {len(test_data)}")

    y_train = [label for _, label in train_data]
    y_test = [label for _, label in test_data]
    classes = ["SPORTS", "POLITICS"]

    all_results = []

    # ---- EXPERIMENT 1: Naive Bayes + Bag of Words ----
    print("\n" + "#" * 55)
    print("  EXPERIMENT 1: Naive Bayes + Bag of Words")
    print("#" * 55)
    vocab_bow = build_vocabulary(train_data, use_ngrams=1)
    print(f"  Vocabulary size (unigrams): {len(vocab_bow)}")

    X_train_bow = [bag_of_words(text, vocab_bow) for text, _ in train_data]
    X_test_bow = [bag_of_words(text, vocab_bow) for text, _ in test_data]

    nb = NaiveBayesClassifier()
    result = run_experiment(nb, "Naive Bayes + BoW", X_train_bow, y_train, X_test_bow, y_test, classes)
    all_results.append(result)

    # ---- EXPERIMENT 2: Logistic Regression + TF-IDF ----
    print("\n" + "#" * 55)
    print("  EXPERIMENT 2: Logistic Regression + TF-IDF")
    print("#" * 55)
    idf_values = compute_idf(train_data, vocab_bow)

    X_train_tfidf = [compute_tfidf(text, vocab_bow, idf_values) for text, _ in train_data]
    X_test_tfidf = [compute_tfidf(text, vocab_bow, idf_values) for text, _ in test_data]

    lr = LogisticRegressionClassifier(lr=0.1, epochs=300)
    result = run_experiment(lr, "Logistic Regression + TF-IDF", X_train_tfidf, y_train, X_test_tfidf, y_test, classes)
    all_results.append(result)

    # ---- EXPERIMENT 3: KNN + Bigram Features ----
    print("\n" + "#" * 55)
    print("  EXPERIMENT 3: KNN + Bigram Features")
    print("#" * 55)
    vocab_ngram = build_vocabulary(train_data, use_ngrams=2)
    print(f"  Vocabulary size (uni+bigrams): {len(vocab_ngram)}")

    X_train_ng = [ngram_features(text, vocab_ngram, n=2) for text, _ in train_data]
    X_test_ng = [ngram_features(text, vocab_ngram, n=2) for text, _ in test_data]

    knn = KNNClassifier(k=5)
    result = run_experiment(knn, "KNN (k=5) + Bigrams", X_train_ng, y_train, X_test_ng, y_test, classes)
    all_results.append(result)

    # ---- FINAL COMPARISON ----
    print("\n" + "=" * 55)
    print("  FINAL COMPARISON")
    print("=" * 55)
    print(f"\n  {'Technique':<35s} {'Accuracy':>10s}")
    print(f"  {'-' * 45}")
    for r in all_results:
        print(f"  {r['name']:<35s}   {r['accuracy'] * 100:.2f}%")

    # find best
    best = max(all_results, key=lambda x: x["accuracy"])
    print(f"\n  Best performing: {best['name']} ({best['accuracy'] * 100:.2f}%)")
    print("=" * 55)


if __name__ == "__main__":
    main()
