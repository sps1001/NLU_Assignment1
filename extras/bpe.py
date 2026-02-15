# Implementing BPE from scratch
# Importing the libraries allowed

import sys
import re
from collections import Counter

# Defining Utility Functions

def read_corpus(path):
    """
    Read corpus from file.
    Each line is treated as a separate sequence.
    We append </w> to mark end of word (classic BPE trick).
    """
    corpus = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                # Split into characters and add end-of-word symbol
                tokens = list(line) + ["</w>"]
                corpus.append(tokens)
    return corpus

def get_pair_frequencies(corpus):
    """
    Count frequency of all adjacent symbol pairs in the corpus.
    """
    pairs = Counter()
    for word in corpus:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] += 1
    return pairs

def merge_pair(pair, corpus):
    """
    Merge a given pair in the entire corpus. Merging occurences of the pairs
    """
    merged_symbol = pair[0] + pair[1]
    new_corpus = []

    for word in corpus:
        new_word = []
        i = 0
        while i < len(word):
            # If we find the pair, merge it
            if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                new_word.append(merged_symbol)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_corpus.append(new_word)

    return new_corpus

def get_vocabulary(corpus):
    """
    Extract the set of symbols from the corpus.
    """
    vocab = set()
    for word in corpus:
        for symbol in word:
            vocab.add(symbol)
    return vocab


# Training the BPE
def main():
    if len(sys.argv) != 2:
        print("Usage: python RollNumber_prob2.py corpus.txt")
        sys.exit(1)

    corpus_path = sys.argv[1]

    # Read corpus
    corpus = read_corpus(corpus_path)

    print("Corpus loaded. Lessgooo")
    print("Number of sequences:", len(corpus))
    print("\n")

    # Ask user for number of merges K
    try:
        K = int(input("Enter number of BPE merges K: "))
    except ValueError:
        print("Please enter a valid integer for K.")
        print("\n")
        sys.exit(1)

    # Initial vocabulary
    vocab = get_vocabulary(corpus)
    print("Initial vocabulary size:", len(vocab))
    print("\n")

    # Perform K merges
    for step in range(K):
        pair_freqs = get_pair_frequencies(corpus)

        if not pair_freqs:
            print("No more pairs to merge.")
            break

        # Find the most frequent pair
        best_pair = max(pair_freqs.items(), key=lambda x: x[1])[0]

        # Merge it
        corpus = merge_pair(best_pair, corpus)

        print(f"Merge {step+1}: Merged pair {best_pair} -> {best_pair[0] + best_pair[1]}")

    # Final vocabulary
    final_vocab = get_vocabulary(corpus)

    print("\nFinal Vocabulary:")
    for token in sorted(final_vocab, key=lambda x: (len(x), x)):
        print(token)

    print("\nFinal vocabulary size:", len(final_vocab))


if __name__ == "__main__":
    main()
