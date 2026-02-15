# Roll Number: B23CS1061


# I have used only the allowed libraries here.
import re
import sys
from collections import Counter, defaultdict



# I have written this function to count the frequency of each word in the corpus,
#and add a </w> marker at end so we know where word ends
def get_word_frequencies(corpus_lines):

    word_freq = Counter()

    for line in corpus_lines:
        # .strip removes whitespaces.
        line = line.strip().lower()
        if not line:
            continue
        # r"[a-zA-Z]+" -> this regex finds all words
        words = re.findall(r"[a-zA-Z]+", line)
        for word in words:
            char_tuple = tuple(list(word) + ["</w>"])
            word_freq[char_tuple] += 1

    return word_freq


# I have written this function to count how often each pair of adjacent symbols appears.
def count_pairs(word_freq):

    pair_counts = defaultdict(int)

    for word, freq in word_freq.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq

    return pair_counts


# I have written this function to merge the most frequent pair into a single new symbol.
def merge_pair(pair, word_freq):

    new_word_freq = {}
    left, right = pair
    merged = left + right  # combining both symbols into one

    for word, freq in word_freq.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == left and word[i + 1] == right:
                new_word.append(merged)
                i += 2  # skipping both since we merged
            else:
                new_word.append(word[i])
                i += 1
        new_word_freq[tuple(new_word)] = freq

    return new_word_freq


# I have written this function to extract all unique symbols from the 
# current word representations.
def get_vocabulary(word_freq):

    vocab = set()
    for word in word_freq.keys():
        for symbol in word:
            vocab.add(symbol)
    return vocab


# This is the main BPE algorithm function.
# Step 1 -> start with character-level vocabulary
# Step 2 -> find the most common adjacent pair and merge it
# Step 3 -> repeat K times or until no more pairs exist
def run_bpe(corpus_lines, num_merges):

 
    word_freq = get_word_frequencies(corpus_lines)

    if not word_freq:
        print("Error: corpus is empty or has no valid words.")
        return

    # printing the initial state so we can see what we started with
    print("=" * 60)
    print("BYTE PAIR ENCODING (BPE) TOKENIZATION")
    print("=" * 60)

    initial_vocab = get_vocabulary(word_freq)
    print(f"\nInitial vocabulary size: {len(initial_vocab)}")
    print(f"Number of merges requested: {num_merges}")
    print(f"Total unique words in corpus: {len(word_freq)}")

    # showing all the initial characters we have
    sorted_initial = sorted(initial_vocab)
    print(f"\nInitial character vocabulary:")
    print(", ".join(sorted_initial))

    print("\n" + "-" * 60)
    print("MERGE OPERATIONS")
    print("-" * 60)

    # now performing K merge operations one by one
    merge_log = []  # keeping track of all merges I did

    for step in range(1, num_merges + 1):
        # counting all adjacent pairs in current vocabulary
        pair_counts = count_pairs(word_freq)

        if not pair_counts:
            # no more pairs left to merge so we stop early
            print(f"\nNo more pairs to merge after {step - 1} merges.")
            break

        # finding the pair with highest frequency
        best_pair = max(pair_counts, key=pair_counts.get)
        best_count = pair_counts[best_pair]

        # printing which pair we are merging in this step
        print(f"\nMerge #{step}: ('{best_pair[0]}', '{best_pair[1]}') -> '{best_pair[0] + best_pair[1]}'  (frequency: {best_count})")

        # merging this pair in all words
        word_freq = merge_pair(best_pair, word_freq)
        merge_log.append((best_pair, best_count))

    # now showing the final results after all merges
    final_vocab = get_vocabulary(word_freq)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nFinal vocabulary size: {len(final_vocab)}")
    sorted_vocab = sorted(final_vocab, key=lambda x: (-len(x), x))
    print(f"\nFinal vocabulary (sorted by length, then alphabetically):")

    # printing each token in the final vocabulary
    for i, token in enumerate(sorted_vocab):
        print(f"  {i + 1}. '{token}'")

    # also showing how words look after all the merges are done
    print(f"\nWord representations after {len(merge_log)} merges:")
    for word, freq in sorted(word_freq.items(), key=lambda x: -x[1]):
        print(f"  {' '.join(word)}  (freq: {freq})")

    print("\n" + "=" * 60)
    print(f"Summary: {len(initial_vocab)} initial symbols -> {len(final_vocab)} symbols after {len(merge_log)} merges")
    print("=" * 60)

    return final_vocab


# This is the main function that handles command line arguments and runs the BPE.
def main():
    # checking if user gave both arguments - k and corpus file
    if len(sys.argv) < 3:
        print("Usage: python b23cs1061_prob2.py k corpus.txt")
        print("  k           - number of BPE merges to perform")
        print("  corpus.txt  - path to the input corpus file")
        sys.exit(1)

    # first argument is the number of merges (k)
    try:
        num_merges = int(sys.argv[1])
    except ValueError:
        print(f"Error: '{sys.argv[1]}' is not a valid number for k.")
        sys.exit(1)

    # second argument is the corpus file path
    corpus_file = sys.argv[2]

    # trying to open and read the corpus file
    try:
        with open(corpus_file, "r") as f:
            corpus_lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: file '{corpus_file}' not found.")
        sys.exit(1)

    print(f"Reading corpus from: {corpus_file}")
    print(f"Total lines read: {len(corpus_lines)}")

    # running the BPE algorithm
    run_bpe(corpus_lines, num_merges)


# this runs the program when executed from terminal
if __name__ == "__main__":
    main()
