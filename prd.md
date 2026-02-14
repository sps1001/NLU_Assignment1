# CSL7640 – Assignment 1
## Product Requirements Document (PRD)

Student: B23CS1061  
Course: Natural Language Understanding  
Deadline: Feb 15, 2026  

---

# 1. Objective

Implement Assignment-1 strictly following constraints:

- No plagiarism
- No external libraries unless explicitly allowed
- All logic implemented from scratch
- Executable from terminal exactly as specified
- Strong inline comments required

---

# 2. Overall Submission Structure

Final submission must be:

rollNumber_A1.zip

Containing:

- RollNumber_prob1.py
- RollNumber_prob1.log
- RollNumber_prob1.txt
- RollNumber_prob2.py
- RollNumber_prob3.py
- RollNumber_prob4.pdf

---

# 3. Problem Specifications

---

## PROBLEM 1 – REGGY++

### Goal
Extend regex-based chatbot using ONLY:

import re  
from datetime import date  

### Functional Requirements

1. Capture full name
   - Extract first name
   - Extract surname (last token)

2. Capture birthday in formats:
   - mm-dd-yy
   - dd-mm-yy
   - dd-mm-yyyy
   - dd Month YYYY
   - dd MonthNameShort YYYY
   - Slash variations also supported

3. Compute age correctly using current date

4. Ask mood
   - Detect positive mood
   - Detect negative mood
   - Handle minor spelling mistakes using regex
   - Provide appropriate responses

5. Log multiple runs
   - Different date formats
   - Mood typos
   - At least one failure case
   - Clear run separation

6. Reflection
   - 300–500 words
   - Discuss:
     - Naturalness
     - Strengths of regex
     - Limitations
     - Failure cases

### Constraints
- Only standard library
- No fuzzy libraries
- Executable:
  python RollNumber_prob1.py

---

## PROBLEM 2 – BYTE PAIR ENCODING

### Goal
Implement BPE from scratch.

### Input
python RollNumber_prob2.py corpus.txt

### Requirements
- Read corpus line by line
- Implement BPE merge algorithm manually
- Use collections, re, sys only
- Return vocabulary after K merges
- No NLP libraries allowed

---

## PROBLEM 3 – NAIVE BAYES

### Goal
Sentiment classifier from scratch.

### Files
pos.txt
neg.txt

### Requirements
- Whitespace tokenization
- Lowercase normalization
- Laplace smoothing
- Train-test split
- Interactive mode after training
- Predict POSITIVE or NEGATIVE

### Execution
python RollNumber_prob3.py

---

## PROBLEM 4 – SPORTS VS POLITICS

### Goal
Document classifier

###
