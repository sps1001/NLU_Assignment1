
# Importing the allowed libraries
import re
from datetime import date

# DEFining a set of Utility Functions for different features of the chatbot

def normalize(text):

    # Lowercase and strip surrounding spaces to reduce variation in input before regex matching.
    return text.strip().lower()


# 1.) Detecting Surname from the given full name

def extract_surname(full_name):
   
    # Assuming that the last sequence of letters as surname
    # This will fail for names having multiple last names
    
    match = re.search(r"([a-zA-Z]+)\s*$", full_name)
    if match:
        return match.group(1)
    return None


# 2.) Function parses the birthdate. 
def parse_birthday(text):
    """
    Tries to parse birthday from multiple possible formats using regex.

    Trying out these 6 combinations :-
    - mm-dd-yy
    - dd-mm-yy
    - dd-mm-yyyy
    - mm-dd-yyyy
    - dd Month YYYY
    - dd Mon YYYY

    Returns:
    - (year, month, day) if successful
    - None if parsing fails
    """

    # Stripping the text 

    text = text.strip()

    # Checking for these 4 patterns first - dd-mm-yyyy or mm-dd-yyyy or dd-mm-yy or mm-dd-yy
    m = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{2}|\d{4})$", text)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        c = int(m.group(3))

        # Planting a Heuristic : If a year has 2 digits, assuming it to be 2000+ for simplicity
        if c < 100:
            c = 2000 + c

        year = c
        # Solving the Ambiguity between dd-mm and mm-dd
        if a > 12 and b <= 12:
            day = a
            month = b
        elif b > 12 and a <= 12:
            day = b
            month = a
        else:
            # If both are <= 12, it's ambiguous -> default to dd-mm
            day = a
            month = b
        
        return (year, month, day)

    # Remaining two combos - dd Month yyyy or dd Mon yyyy
    m = re.match(r"^(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})$", text)
    if m:
        day = int(m.group(1))
        month_name = m.group(2).lower()
        year = int(m.group(3))

        # Map month names to numbers
        months = {
            "jan": 1, "january": 1,
            "feb": 2, "february": 2,
            "mar": 3, "march": 3,
            "apr": 4, "april": 4,
            "may": 5,
            "jun": 6, "june": 6,
            "jul": 7, "july": 7,
            "aug": 8, "august": 8,
            "sep": 9, "september": 9,
            "oct": 10, "october": 10,
            "nov": 11, "november": 11,
            "dec": 12, "december": 12,
        }

        # Match by prefix Like "aug" for "August"
        for name, num in months.items():
            if re.match(name, month_name):
                return (year, num, day)

    return None

# A function to calculate the age till today's date from the birth-date
def calculate_age(year, month, day):
    today = date.today()
    age = today.year - year

    # Subtracting by 1 if the birthcay for this year hasnt occurred yet !
    if (today.month, today.day) < (month, day):
        age -= 1

    return age

# A function to format the birthday
def format_birthday(year, month, day):

    # Day suffix logic - 1st / 2nd / 3rd / 4th etc. 
    if 11 <= day <= 13:
        suffix = "th"
    else:
        if day % 10 == 1:
            suffix = "st"
        elif day % 10 == 2:
            suffix = "nd"
        elif day % 10 == 3:
            suffix = "rd"
        else:
            suffix = "th"

    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }

    month_str = month_names.get(month, "Unknown")
    return f"{day}{suffix} {month_str} {year}"


# A function to detect the mood using regex. 
def detect_mood(text):
   
    text = normalize(text)

    # Checking Happy , Sad , Angry , Tired , Stressed , Anxious , Bored , Confused and Sick Patterns with some tolerance to spelling errors
    if re.search(r"(hap+|hapy|happy|gud|good|great|awes|nice|excite|yay)", text):
        return "happy"

    if re.search(r"(sad+|sed|bad|down|upset|depress)", text):
        return "sad"

    if re.search(r"(angr|mad|furious|annoy|irritat)", text):
        return "angry"

    if re.search(r"(tired|sleep|exhaust|fatig)", text):
        return "tired"

    if re.search(r"(stress|pressur|overwhelm)", text):
        return "stressed"

    if re.search(r"(anx|nervous|worri)", text):
        return "anxious"

    if re.search(r"(bore|boring|meh)", text):
        return "bored"

    if re.search(r"(confus|lost|unsure)", text):
        return "confused"

    if re.search(r"(sick|ill|fever|cold)", text):
        return "sick"

    return "unknown"


# The Main Chatbot Loop 
def main():
    print("Reggy++: Greetings! I'm a regex-based chatbot.")
    print("Reggy++: What's your full name?")

    # Asks for my name
    name = input("You: ")
    surname = extract_surname(name)

    if surname:
        print(f"Reggy++: Nice to meet you, {surname}!")
    else:
        print("Reggy++: Nice to meet you!")

    # Ask for my birthday
    print("Reggy++: What's your birthdate?")

    bday_input = input("You: ")
    parsed = parse_birthday(bday_input)

    if parsed is None:
        print("Reggy++: i am sorry but, I couldn't understand that date format.")
        print("Reggy++: As a regex-based chatbot only, this is my limitation.")
    else:
        year, month, day = parsed
        try:
            age = calculate_age(year, month, day)
            formatted = format_birthday(year, month, day)
            print(f"Reggy++: Your birthday is on {formatted}.")
            print(f"Reggy++: Also,you are approximately {age} years old.")
        except Exception:
            print("Reggy++: That date seems invalid.")

    # Ask for my mood
    print("Reggy++: How are you feeling today?")

    mood_input = input("You: ")
    mood = detect_mood(mood_input)

    if mood == "happy":
        print("Reggy++: That's awesome! I love the positive vibes 😄")
    elif mood == "sad":
        print("Reggy++: I'm sorry you're feeling that way. Hope things improve soon 💛")
    elif mood == "angry":
        print("Reggy++: Oof, sounds rough. Maybe take a deep breath.")
    elif mood == "tired":
        print("Reggy++: You should really get some rest. Your body will thank you!")
    elif mood == "stressed":
        print("Reggy++: Stress can be tough. Try to take things one step at a time.")
    elif mood == "anxious":
        print("Reggy++: That sounds uncomfortable. You're not alone in feeling this way.")
    elif mood == "bored":
        print("Reggy++: Boredom detected! Maybe try something new today?")
    elif mood == "confused":
        print("Reggy++: Yeah, that happens sometimes. Things will probably clear up.")
    elif mood == "sick":
        print("Reggy++: Oh no! Take care and get well soon 🤒")
    else:
        print("Reggy++: Hmm, I can't really tell how you're feeling. But I hope you're okay!")

    print("Reggy++: Goodbye!")



if __name__ == "__main__":
    main()
