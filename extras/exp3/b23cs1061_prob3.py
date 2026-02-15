# Roll Number: B23CS1061
# Program reads pos.txt and neg.txt from the same directory,


# I have used only the allowed standard library imports here.
import os
import random
import math


# I have written this function to split a sentence into words.
# .strip() removes whitespace, .lower() converts to lowercase, .split() breaks by spaces.
def tokenize(sentence):
    words = sentence.strip().lower().split()
    return words


# I have written this function to read the data from a file.
# each line in the file is one sentence.
def load_data(filepath):
    sentences = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # skipping empty lines
                sentences.append(line)
    return sentences


# I have written this function to shuffle and split data into train and test sets.
# using 80-20 split with a fixed seed so results are same every time.
def split_data(sentences, train_ratio=0.8):
    shuffled = sentences[:]
    random.seed(42)  # fixed seed for reproducibility
    random.shuffle(shuffled)
    split_point = int(len(shuffled) * train_ratio)
    return shuffled[:split_point], shuffled[split_point:]


# I have written this function to train the Naive Bayes model.
# it calculates P(positive) and P(negative) as prior probabilities
# and P(word | class) for each word as likelihood with Laplace smoothing.
def train_naive_bayes(pos_train, neg_train):

    # counting total documents in each class
    total_docs = len(pos_train) + len(neg_train)
    prior_pos = len(pos_train) / total_docs
    prior_neg = len(neg_train) / total_docs

    # these dictionaries will store word frequencies for each class
    pos_word_counts = {}
    neg_word_counts = {}
    pos_total_words = 0
    neg_total_words = 0

    # building vocabulary which is all unique words across both classes
    vocab = set()

    # counting words in positive sentences
    for sentence in pos_train:
        words = tokenize(sentence)
        for w in words:
            vocab.add(w)
            pos_word_counts[w] = pos_word_counts.get(w, 0) + 1
            pos_total_words += 1

    # counting words in negative sentences
    for sentence in neg_train:
        words = tokenize(sentence)
        for w in words:
            vocab.add(w)
            neg_word_counts[w] = neg_word_counts.get(w, 0) + 1
            neg_total_words += 1

    vocab_size = len(vocab)

    # storing everything needed for prediction in a dictionary
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


# I have written this function to predict sentiment of a sentence.
# it uses log probabilities to avoid underflow with many multiplications.
# Laplace smoothing formula: P(word|class) = (count + 1) / (total + vocab_size)
def predict(model, sentence):
    words = tokenize(sentence)

    # starting with log of prior probabilities
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


# I have written this function to test the model on held-out data and print accuracy.
def evaluate(model, test_pos, test_neg):
    correct = 0
    total = 0

    # testing on positive sentences
    for sentence in test_pos:
        label, _, _ = predict(model, sentence)
        if label == "POSITIVE":
            correct += 1
        total += 1

    # testing on negative sentences
    for sentence in test_neg:
        label, _, _ = predict(model, sentence)
        if label == "NEGATIVE":
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return correct, total, accuracy


# This is the main function that loads data, trains model, evaluates and runs interactive mode.
def main():
    # finding the directory where this script is located
    # so I can find pos.txt and neg.txt in the same folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pos_file = os.path.join(script_dir, "pos.txt")
    neg_file = os.path.join(script_dir, "neg.txt")

    # loading data from both files
    print("Loading data...")
    pos_sentences = load_data(pos_file)
    neg_sentences = load_data(neg_file)
    print(f"  Positive sentences: {len(pos_sentences)}")
    print(f"  Negative sentences: {len(neg_sentences)}")

    # splitting into train and test with 80-20 ratio
    pos_train, pos_test = split_data(pos_sentences, train_ratio=0.8)
    neg_train, neg_test = split_data(neg_sentences, train_ratio=0.8)
    print(f"\nTraining set: {len(pos_train)} positive, {len(neg_train)} negative")
    print(f"Test set:     {len(pos_test)} positive, {len(neg_test)} negative")

    # training the model
    print("\nTraining Naive Bayes model...")
    model = train_naive_bayes(pos_train, neg_train)
    print(f"  Vocabulary size: {model['vocab_size']}")
    print(f"  Prior P(positive): {model['prior_pos']:.4f}")
    print(f"  Prior P(negative): {model['prior_neg']:.4f}")
    print("Training complete!")

    # evaluating on the test set
    correct, total, accuracy = evaluate(model, pos_test, neg_test)
    print(f"\nTest Accuracy: {correct}/{total} = {accuracy * 100:.2f}%")

    # interactive mode where user can type sentences and get predictions
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


# this runs the program when executed from terminal
if __name__ == "__main__":
    main()
