import math
import random
from collections import defaultdict, Counter

# -----------------------------
# Utility functions
# -----------------------------

def load_sentences(filename):
    sentences = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
    return sentences

def tokenize(sentence):
    # Simple whitespace tokenization + lowercase
    return sentence.lower().split()

# -----------------------------
# Naive Bayes Classifier
# -----------------------------

class NaiveBayesClassifier:
    def __init__(self):
        self.vocab = set()
        self.word_counts = {
            "pos": Counter(),
            "neg": Counter()
        }
        self.class_counts = {
            "pos": 0,
            "neg": 0
        }
        self.total_words = {
            "pos": 0,
            "neg": 0
        }
        self.priors = {
            "pos": 0.0,
            "neg": 0.0
        }

    def train(self, pos_sentences, neg_sentences):
        # Count documents
        num_pos = len(pos_sentences)
        num_neg = len(neg_sentences)
        total_docs = num_pos + num_neg

        # Compute priors
        self.priors["pos"] = math.log(num_pos / total_docs)
        self.priors["neg"] = math.log(num_neg / total_docs)

        # Process positive sentences
        for sent in pos_sentences:
            self.class_counts["pos"] += 1
            tokens = tokenize(sent)
            for word in tokens:
                self.vocab.add(word)
                self.word_counts["pos"][word] += 1
                self.total_words["pos"] += 1

        # Process negative sentences
        for sent in neg_sentences:
            self.class_counts["neg"] += 1
            tokens = tokenize(sent)
            for word in tokens:
                self.vocab.add(word)
                self.word_counts["neg"][word] += 1
                self.total_words["neg"] += 1

    def predict(self, sentence):
        tokens = tokenize(sentence)

        # Start with log priors
        log_prob_pos = self.priors["pos"]
        log_prob_neg = self.priors["neg"]

        V = len(self.vocab)  # vocabulary size

        for word in tokens:
            # Laplace smoothing
            pos_count = self.word_counts["pos"][word]
            neg_count = self.word_counts["neg"][word]

            prob_word_pos = (pos_count + 1) / (self.total_words["pos"] + V)
            prob_word_neg = (neg_count + 1) / (self.total_words["neg"] + V)

            log_prob_pos += math.log(prob_word_pos)
            log_prob_neg += math.log(prob_word_neg)

        if log_prob_pos >= log_prob_neg:
            return "POSITIVE"
        else:
            return "NEGATIVE"

# -----------------------------
# Train / Validation Split
# -----------------------------

def train_validation_split(data, split_ratio=0.8):
    random.shuffle(data)
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    val_data = data[split_index:]
    return train_data, val_data

# -----------------------------
# Main Program
# -----------------------------

def main():
    # Load data
    pos_sentences = load_sentences("pos.txt")
    neg_sentences = load_sentences("neg.txt")

    if len(pos_sentences) == 0 or len(neg_sentences) == 0:
        print("Error: pos.txt or neg.txt is empty or missing.")
        return

    # Split into train and validation
    pos_train, pos_val = train_validation_split(pos_sentences, 0.8)
    neg_train, neg_val = train_validation_split(neg_sentences, 0.8)

    # Train classifier
    nb = NaiveBayesClassifier()
    nb.train(pos_train, neg_train)

    # Validate
    correct = 0
    total = 0

    for sent in pos_val:
        pred = nb.predict(sent)
        if pred == "POSITIVE":
            correct += 1
        total += 1

    for sent in neg_val:
        pred = nb.predict(sent)
        if pred == "NEGATIVE":
            correct += 1
        total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Validation Accuracy: {accuracy:.2f}%")

    # Interactive mode
    print("\nEnter a sentence to classify sentiment (type 'exit' to quit):")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        if not user_input:
            continue
        prediction = nb.predict(user_input)
        print("Prediction:", prediction)

if __name__ == "__main__":
    main()
