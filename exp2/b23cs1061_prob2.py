# b23cs1061_prob2.py
# Byte Pair Encoding (BPE) Tokenization - implemented from scratch
# Roll Number: B23CS1061
# Course: CSL7640 - Natural Language Understanding
#
# Usage: python b23cs1061_prob2.py corpus.txt
# The program reads a corpus file and applies BPE merges, then prints the final vocabulary.

import re
import sys
from collections import Counter, defaultdict


def get_word_frequencies(corpus_lines):
    """
    Read the corpus and count how often each word appears.
    Each word is split into individual characters with a special </w> end-of-word marker.
    For example, "low" appears 5 times => ('l', 'o', 'w', '</w>'): 5
    """
    word_freq = Counter()

    for line in corpus_lines:
        line = line.strip().lower()
        if not line:
            continue
        # split line into words using whitespace
        words = re.findall(r"[a-zA-Z]+", line)
        for word in words:
            # represent each word as tuple of characters + end marker
            # the end-of-word marker helps us reconstruct words later
            char_tuple = tuple(list(word) + ["</w>"])
            word_freq[char_tuple] += 1

    return word_freq


def count_pairs(word_freq):
    """
    Count frequency of all adjacent symbol pairs across the vocabulary.
    For example, if word ('l','o','w','</w>') has freq 5,
    then pairs ('l','o'), ('o','w'), ('w','</w>') each get +5.
    """
    pair_counts = defaultdict(int)

    for word, freq in word_freq.items():
        # go through consecutive pairs in the word
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq

    return pair_counts


def merge_pair(pair, word_freq):
    """
    Merge the most frequent pair into a single symbol in all words.
    For example, merging ('e','s') turns ('n','e','w','e','s','t','</w>')
    into ('n','e','w','es','t','</w>').
    """
    new_word_freq = {}
    left, right = pair
    merged = left + right  # the new combined symbol

    for word, freq in word_freq.items():
        new_word = []
        i = 0
        while i < len(word):
            # check if current position matches the pair we want to merge
            if i < len(word) - 1 and word[i] == left and word[i + 1] == right:
                new_word.append(merged)
                i += 2  # skip both symbols since we merged them
            else:
                new_word.append(word[i])
                i += 1
        new_word_freq[tuple(new_word)] = freq

    return new_word_freq


def get_vocabulary(word_freq):
    """
    Extract all unique symbols (subwords) from the current word representations.
    This gives us our BPE vocabulary at any point during training.
    """
    vocab = set()
    for word in word_freq.keys():
        for symbol in word:
            vocab.add(symbol)
    return vocab


def run_bpe(corpus_lines, num_merges):
    """
    Main BPE algorithm:
    1. Start with character-level vocabulary
    2. Repeatedly find the most common adjacent pair
    3. Merge that pair into a new symbol
    4. Repeat K times (or until no more pairs exist)
    """
    # Step 1: get initial word frequencies as character tuples
    word_freq = get_word_frequencies(corpus_lines)

    if not word_freq:
        print("Error: corpus is empty or has no valid words.")
        return

    # print initial state
    print("=" * 60)
    print("BYTE PAIR ENCODING (BPE) TOKENIZATION")
    print("=" * 60)

    initial_vocab = get_vocabulary(word_freq)
    print(f"\nInitial vocabulary size: {len(initial_vocab)}")
    print(f"Number of merges requested: {num_merges}")
    print(f"Total unique words in corpus: {len(word_freq)}")

    # show initial character vocabulary
    sorted_initial = sorted(initial_vocab)
    print(f"\nInitial character vocabulary:")
    print(", ".join(sorted_initial))

    print("\n" + "-" * 60)
    print("MERGE OPERATIONS")
    print("-" * 60)

    # Step 2: perform K merge operations
    merge_log = []  # keep track of all merges we did

    for step in range(1, num_merges + 1):
        # count all adjacent pairs
        pair_counts = count_pairs(word_freq)

        if not pair_counts:
            print(f"\nNo more pairs to merge after {step - 1} merges.")
            break

        # find the most frequent pair
        best_pair = max(pair_counts, key=pair_counts.get)
        best_count = pair_counts[best_pair]

        # show what we're merging
        print(f"\nMerge #{step}: ('{best_pair[0]}', '{best_pair[1]}') -> '{best_pair[0] + best_pair[1]}'  (frequency: {best_count})")

        # merge the best pair in all words
        word_freq = merge_pair(best_pair, word_freq)
        merge_log.append((best_pair, best_count))

    # Step 3: get and display the final vocabulary
    final_vocab = get_vocabulary(word_freq)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nFinal vocabulary size: {len(final_vocab)}")
    sorted_vocab = sorted(final_vocab, key=lambda x: (-len(x), x))
    print(f"\nFinal vocabulary (sorted by length, then alphabetically):")

    # print vocabulary in a readable way
    for i, token in enumerate(sorted_vocab):
        print(f"  {i + 1}. '{token}'")

    # also show how words look after all merges
    print(f"\nWord representations after {len(merge_log)} merges:")
    for word, freq in sorted(word_freq.items(), key=lambda x: -x[1]):
        print(f"  {' '.join(word)}  (freq: {freq})")

    print("\n" + "=" * 60)
    print(f"Summary: {len(initial_vocab)} initial symbols -> {len(final_vocab)} symbols after {len(merge_log)} merges")
    print("=" * 60)

    return final_vocab


def main():
    # check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python b23cs1061_prob2.py corpus.txt [num_merges]")
        print("  corpus.txt  - path to the input corpus file")
        print("  num_merges  - number of BPE merges to perform (default: 10)")
        sys.exit(1)

    corpus_file = sys.argv[1]

    # optional: let user specify number of merges (default 10)
    num_merges = 10
    if len(sys.argv) >= 3:
        try:
            num_merges = int(sys.argv[2])
        except ValueError:
            print(f"Error: '{sys.argv[2]}' is not a valid number for merges.")
            sys.exit(1)

    # read the corpus file
    try:
        with open(corpus_file, "r") as f:
            corpus_lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: file '{corpus_file}' not found.")
        sys.exit(1)

    print(f"Reading corpus from: {corpus_file}")
    print(f"Total lines read: {len(corpus_lines)}")

    # run BPE
    run_bpe(corpus_lines, num_merges)


if __name__ == "__main__":
    main()
