# b23cs1061_prob3.py
# Naive Bayes Sentiment Classifier - implemented from scratch
# Roll Number: B23CS1061
# Course: CSL7640 - Natural Language Understanding
#
# Usage: python b23cs1061_prob3.py
# Reads pos.txt and neg.txt from the same directory,
# trains a Naive Bayes model, then enters interactive prediction mode.

import os
import random
import math


# ---- Tokenization ----
# simple whitespace split + lowercase, as required
def tokenize(sentence):
    words = sentence.strip().lower().split()
    return words


# ---- Data Loading ----
def load_data(filepath):
    """Read file line by line, each line is one sentence."""
    sentences = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                sentences.append(line)
    return sentences


# ---- Train/Test Split ----
def split_data(sentences, train_ratio=0.8):
    """Shuffle and split data into train and test sets."""
    shuffled = sentences[:]
    random.seed(42)  # fixed seed so results are reproducible
    random.shuffle(shuffled)
    split_point = int(len(shuffled) * train_ratio)
    return shuffled[:split_point], shuffled[split_point:]


# ---- Naive Bayes Training ----
def train_naive_bayes(pos_train, neg_train):
    """
    Train the Naive Bayes model.
    We need:
      - P(positive) and P(negative) => prior probabilities
      - P(word | class) for each word => likelihood with Laplace smoothing
    """
    # count total documents in each class
    total_docs = len(pos_train) + len(neg_train)
    prior_pos = len(pos_train) / total_docs
    prior_neg = len(neg_train) / total_docs

    # count word frequencies in each class
    pos_word_counts = {}
    neg_word_counts = {}
    pos_total_words = 0
    neg_total_words = 0

    # build vocabulary (all unique words across both classes)
    vocab = set()

    # count words in positive sentences
    for sentence in pos_train:
        words = tokenize(sentence)
        for w in words:
            vocab.add(w)
            pos_word_counts[w] = pos_word_counts.get(w, 0) + 1
            pos_total_words += 1

    # count words in negative sentences
    for sentence in neg_train:
        words = tokenize(sentence)
        for w in words:
            vocab.add(w)
            neg_word_counts[w] = neg_word_counts.get(w, 0) + 1
            neg_total_words += 1

    vocab_size = len(vocab)

    # store everything we need for prediction in a dictionary
    model = {
        "prior_pos": prior_pos,
        "prior_neg": prior_neg,
        "pos_word_counts": pos_word_counts,
        "neg_word_counts": neg_word_counts,
        "pos_total_words": pos_total_words,
        "neg_total_words": neg_total_words,
        "vocab_size": vocab_size,
        "vocab": vocab
    }

    return model


# ---- Prediction ----
def predict(model, sentence):
    """
    Predict sentiment of a sentence using trained Naive Bayes model.
    Uses log probabilities to avoid underflow with many multiplications.
    Applies Laplace smoothing: P(word|class) = (count + 1) / (total + vocab_size)
    """
    words = tokenize(sentence)

    # start with log of prior probabilities
    log_prob_pos = math.log(model["prior_pos"])
    log_prob_neg = math.log(model["prior_neg"])

    V = model["vocab_size"]

    for w in words:
        # Laplace smoothed probability for positive class
        count_pos = model["pos_word_counts"].get(w, 0)
        prob_w_pos = (count_pos + 1) / (model["pos_total_words"] + V)
        log_prob_pos += math.log(prob_w_pos)

        # Laplace smoothed probability for negative class
        count_neg = model["neg_word_counts"].get(w, 0)
        prob_w_neg = (count_neg + 1) / (model["neg_total_words"] + V)
        log_prob_neg += math.log(prob_w_neg)

    # whichever class has higher log-probability wins
    if log_prob_pos >= log_prob_neg:
        return "POSITIVE", log_prob_pos, log_prob_neg
    else:
        return "NEGATIVE", log_prob_pos, log_prob_neg


# ---- Evaluation ----
def evaluate(model, test_pos, test_neg):
    """Test the model on held-out data and print accuracy."""
    correct = 0
    total = 0

    # test on positive sentences
    for sentence in test_pos:
        label, _, _ = predict(model, sentence)
        if label == "POSITIVE":
            correct += 1
        total += 1

    # test on negative sentences
    for sentence in test_neg:
        label, _, _ = predict(model, sentence)
        if label == "NEGATIVE":
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return correct, total, accuracy


# ---- Main ----
def main():
    # figure out directory where this script is located
    # so we can find pos.txt and neg.txt in the same folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pos_file = os.path.join(script_dir, "pos.txt")
    neg_file = os.path.join(script_dir, "neg.txt")

    # load data
    print("Loading data...")
    pos_sentences = load_data(pos_file)
    neg_sentences = load_data(neg_file)
    print(f"  Positive sentences: {len(pos_sentences)}")
    print(f"  Negative sentences: {len(neg_sentences)}")

    # split into train and test (80-20 split)
    pos_train, pos_test = split_data(pos_sentences, train_ratio=0.8)
    neg_train, neg_test = split_data(neg_sentences, train_ratio=0.8)
    print(f"\nTraining set: {len(pos_train)} positive, {len(neg_train)} negative")
    print(f"Test set:     {len(pos_test)} positive, {len(neg_test)} negative")

    # train the model
    print("\nTraining Naive Bayes model...")
    model = train_naive_bayes(pos_train, neg_train)
    print(f"  Vocabulary size: {model['vocab_size']}")
    print(f"  Prior P(positive): {model['prior_pos']:.4f}")
    print(f"  Prior P(negative): {model['prior_neg']:.4f}")
    print("Training complete!")

    # evaluate on test set
    correct, total, accuracy = evaluate(model, pos_test, neg_test)
    print(f"\nTest Accuracy: {correct}/{total} = {accuracy * 100:.2f}%")

    # interactive mode
    print("\n" + "=" * 50)
    print("INTERACTIVE SENTIMENT PREDICTION")
    print("=" * 50)
    print("Enter a sentence to classify (type 'quit' to exit):\n")

    while True:
        try:
            user_input = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        label, log_pos, log_neg = predict(model, user_input)
        print(f"Prediction: {label}")
        print(f"  (log P(pos) = {log_pos:.4f}, log P(neg) = {log_neg:.4f})")
        print()


if __name__ == "__main__":
    main()
