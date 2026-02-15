import difflib

file1_lines = open('exp3/b23cs1061_prob3.py').readlines()
file2_lines = open('pos_neg.py').readlines()

# Get a similarity ratio (ranges from 0.0 to 1.0)
matcher = difflib.SequenceMatcher(None, file1_lines, file2_lines)
similarity_ratio = matcher.ratio()
print(f"The similarity ratio is: {similarity_ratio:.2%}")

# Generate a human-readable diff
differ = difflib.Differ()
diffs = list(differ.compare(file1_lines, file2_lines))
# print(''.join(diffs)) # Uncomment to print the detailed differences
