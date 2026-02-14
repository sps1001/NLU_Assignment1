# b23cs1061_prob1.py
# Reggy++ : A regex-based chatbot for NLU Assignment 1
# Roll Number: B23CS1061

import re
from datetime import date


# Function to compute age from birthday
def calculate_age(day, month, year):
    today = date.today()
    age = today.year - year
    # if birthday hasn't come yet this year, subtract 1
    if (today.month, today.day) < (month, day):
        age -= 1
    return age


# Function to parse birthday from user input
# Supports: dd-mm-yyyy, dd-mm-yy, mm/dd/yy, dd Month YYYY, dd Mon YYYY
def parse_birthday(text):
    text = text.strip()

    # Try numeric format first: dd-mm-yyyy or dd/mm/yy etc.
    match = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})\b", text)
    if match:
        d, m, y = match.groups()
        d, m, y = int(d), int(m), int(y)
        # handle two-digit year
        if y < 100:
            y += 2000 if y < 26 else 1900
        return d, m, y

    # Try textual format: dd Month yyyy (full or short month name)
    months_map = {
        "january": 1, "jan": 1, "february": 2, "feb": 2,
        "march": 3, "mar": 3, "april": 4, "apr": 4,
        "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
        "august": 8, "aug": 8, "september": 9, "sep": 9,
        "october": 10, "oct": 10, "november": 11, "nov": 11,
        "december": 12, "dec": 12
    }

    match = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b", text)
    if match:
        d, month_str, y = match.groups()
        month_num = months_map.get(month_str.lower())
        if month_num:
            return int(d), month_num, int(y)

    return None


# Function to detect mood using regex with common misspellings
def detect_mood(text):
    text = text.lower()

    # positive moods including common typos
    positive = r"\b(ha+p+y|hapy|good|gud|grt|great|grate|awesome|awsome|fine|fyn|wonderful|excited|exited|fantastic|ok|okay)\b"
    # negative moods including common typos
    negative = r"\b(sa+d|saad|bad|baad|tired|tyred|tird|angry|angri|angery|stressed|stressd|depressed|depresed|lonely|lonley|bored|borred|anxious|anxios)\b"

    if re.search(positive, text):
        return "good"
    if re.search(negative, text):
        return "bad"
    return "neutral"


# Main chatbot function
def chatbot():
    # store user info as we go
    user_name = None
    user_surname = None
    user_age = None

    print("Reggy: Hi! What is your full name?")

    while True:
        user_input = input("You: ").strip()

        # check if user wants to exit at any point
        if re.search(r"\b(exit|quit|bye|goodbye)\b", user_input.lower()):
            first = user_name if user_name else "friend"
            print(f"Reggy: Goodbye {first}!")
            break

        # Step 1: get the name first
        if user_name is None:
            # split name into parts to extract first name and surname
            parts = user_input.split()
            if len(parts) >= 1:
                user_name = parts[0]
            if len(parts) > 1:
                user_surname = parts[-1]  # last word is surname
                print(f"Reggy: Nice to meet you {user_name} {user_surname}!")
            else:
                print(f"Reggy: Nice to meet you {user_name}!")
            print("Reggy: When is your birthday?")
            continue

        # Step 2: get birthday and calculate age
        if user_age is None:
            result = parse_birthday(user_input)
            if result:
                d, m, y = result
                user_age = calculate_age(d, m, y)
                print(f"Reggy: You are {user_age} years old!")
                print("Reggy: How are you feeling today?")
            else:
                print("Reggy: I couldn't understand that date. Please try again.")
            continue

        # Step 3: mood detection and response
        mood = detect_mood(user_input)
        if mood == "good":
            print("Reggy: That's wonderful! Keep smiling!")
        elif mood == "bad":
            print("Reggy: I'm sorry to hear that. Things will get better!")
        else:
            print("Reggy: I see. Tell me more about it.")


# run the chatbot when file is executed
if __name__ == "__main__":
    chatbot()
