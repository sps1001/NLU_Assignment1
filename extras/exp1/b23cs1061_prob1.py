# Roll Number: B23CS1061

# I have used only the allowed 2 imports here as told by sir.
import re
from datetime import date


# This function is basically doing 2 things:-
# first .strip() is removing all unnecessary white spaces
# second .lower() converts all chars to lowercase.
def clean_input(text):
    
    return text.strip().lower()



# I'm treating the last word in the name as the surname
#[a-zA-Z]+)\s*$-> This regex match one or more letter at end of the word (mostly surnames)
def get_surname(full_name):
    # match.group(1) gives the surname
    match = re.search(r"([a-zA-Z]+)\s*$", full_name.strip())
    if match:
        return match.group(1)
    return None


#I have created this func to ask for birthdate in specific formats and then return the year, month and day.
def parse_birthday(text):
    """
    Supported date formats:
    - dd-mm-yyyy or mm-dd-yyyy (with - or / separator)
    - dd-mm-yy or mm-dd-yy
    - dd MonthName yyyy (e.g. 21 September 2004)
    - dd Mon yyyy (e.g. 21 Sep 2004)

    Returns (year, month, day) or None if unable to parse
    """
    text = text.strip()

    # ^(\d{1,2})[-/](\d{1,2})[-/](\d{2}|\d{4})$ -> This regex is first trying the patterns like dd-mm-yyyy, mm/dd/yy, etc
    m = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{2}|\d{4})$", text)
    if m:
        first_num = int(m.group(1))
        second_num = int(m.group(2))
        year_part = int(m.group(3))

        # if year is 2 digit, I am adding 2000 in it to make it 4 digit 
        if year_part < 100:
            year_part = 2000 + year_part

        year = year_part

        # This if-else decide whether the format is dd-mm or mm-dd
        # if a number > 12 it must be a day..
        if first_num > 12 and second_num <= 12:
            day = first_num
            month = second_num
        elif second_num > 12 and first_num <= 12:
            day = second_num
            month = first_num
        else:
            # if both are less than or equal to 12, then it is ambiguous, so use only dd-mmy format.
            day = first_num
            month = second_num

        return (year, month, day)

    # ^(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})$ -> now I am trying text format like "21 September 2004" or "21 Sep 2004"
    m = re.match(r"^(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})$", text)
    if m:
        day = int(m.group(1))
        month_text = m.group(2).lower()
        year = int(m.group(3))

        # creating a dictionary to map month names to numbers   
        month_lookup = {
            "jan": 1, "january": 1,
            "feb": 2, "february": 2,
            "mar": 3, "march": 3,
            "apr": 4, "april": 4,
            "may": 5, "may":5,
            "jun": 6, "june": 6,
            "jul": 7, "july": 7,
            "aug": 8, "august": 8,
            "sep": 9, "september": 9,
            "oct": 10, "october": 10,
            "nov": 11, "november": 11,
            "dec": 12, "december": 12,
        }

        # try to match the month name in dictionary also handles partial matches like "sept"
        for name, num in month_lookup.items():
            if re.match(name, month_text):
                return (year, num, day)

    # if nothing matched, return None
    return None


# I have created this function to calculate age from year,month and day.
def calculate_age(year, month, day):
    today = date.today()
    age = today.year - year

    # if birthday hasn't happened yet this year, subtract 1
    if (today.month, today.day) < (month, day):
        age -= 1

    return age


# This function formats the date look nice for printing - like "21st September 2004" instead of just numbers
def format_date(year, month, day):
    
    if 11 <= day <= 13:
        suffix = "th"  
    elif day % 10 == 1:
        suffix = "st"
    elif day % 10 == 2:
        suffix = "nd"
    elif day % 10 == 3:
        suffix = "rd"
    else:
        suffix = "th"

    # dictionary to map month numbers to names  
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }

    return f"{day}{suffix} {month_names.get(month, 'Unknown')} {year}"


# This function Detect the mood from what the user says
def detect_mood(text):
    text = clean_input(text)

  
    if re.search(r"(go+d|gud|hap+y|hapy|great|well|wll)", text):
        return "mood is happy"
    if re.search(r"(bad|sad|tired|sick|ill|fever|cold|unwell|nausea|angry|stressed|down|upset|depres|unhappy)", text):
        return "mood is sad"

   
    if re.search(r"(angr|mad|furious|annoy|irritat|frustrat)", text):
        return "mood is angry"

   
    if re.search(r"(ok|okay|fine|alright|decent|normal)", text):
        return "okay"


    return "unknown"


# The main chatbot loop which is handling all the conversation.
def main():
    print("Reggy++: Hey there! I'm Reggy++, a regex-powered chatbot for assignment 1")
    print("Reggy++: We can start a chat now. Tell me your full name?")

    # 1.) Get user name input and strip removes whitespaces.
    name_input = input("You: ").strip()

    # 2.) using regex exp to get the surname from earlier defined function.
    surname = get_surname(name_input)
    # first name is just the first word
    first_name = name_input.split()[0] if name_input.split() else name_input

    if surname and surname.lower() != first_name.lower():
        print(f"Reggy++: Hello {first_name}! I predict, your surname is {surname}.")
    else:
        print(f"Reggy++: Hello {first_name}!")

    # Step 2 - ask for birthday and calculate age
    print("Reggy++: Tell me your DOB? (you can type it in any format like dd-mm-yyyy, dd/mm/yy, or 21 Sep 2004)")

    bday_input = input("You: ").strip()
    parsed = parse_birthday(bday_input)

    if parsed is None:
        # if it is none, user entered somethign that is not defined.
        print("Reggy++: Hmm, I couldn't parse that date format. That's one of my limitations as a regex bot!")
        print("Reggy++: Try something like 21-09-2004 or 21 September 2004 next time.")
    else:
        year, month, day = parsed
        try:
            age = calculate_age(year, month, day)
            date_birth = format_date(year, month, day)
            print(f"Reggy++: So your birthday is on {date_birth}.")
            print(f"Reggy++: And so you are {age} years old!")
        except Exception:
            print("Reggy++: That date doesn't seem right. Could you check it?")

    # Step 3 - mood detection and conversation loop
    # keeps running until the user types bye, quit or exit
    print("Reggy++: Anyway, how are you feeling right now?, If you don't want to talk -> type ( bye | quit |exit )")
    
    # I have added these as flag so that chatbot always doesnot say the same thign again and again..
    a = 1
    b = 0

    while True:
        mood_input = input("You: ").strip()

        # check if user wants to leave
        if re.search(r"\b(bye|quit|exit|goodbye)\b", mood_input.lower()):
            print(f"Reggy++: It was nice talking to you, {first_name}. See you after Minors! GoodLuck!")
            break

        mood = detect_mood(mood_input)

        # giving different responses based on what mood was detected
        if mood == "mood is happy":
            print("Reggy++: That's great to hear! Keep the positive energy going!")
            print("Reggy++: Man That is interesting, we should talk about it more or if you are busy just type quit!!")
            
        elif mood == "mood is sad":
            print("Reggy++: I'm sorry to hear that. Hope things get better for you soon.")
            print("Reggy++: Man That is interesting, we should talk about it more or if you are busy just type quit!!")
        elif mood == "mood is angry":
            print("Reggy++: Take a deep breath. Sometimes stepping away helps.")    
            print("Reggy++: Man That is interesting, we should talk about it more or if you are busy just type quit!!")
        elif mood == "okay":
            print("Reggy++: That's alright! Not every day has to be amazing.")
            print("Reggy++: If you want to talk more, tell me something else simply type quit, we will talk another day. ")
        else:
            # Taken from original notebook shared by sir for reference 
            match = re.search(r"i am (.*)|i'm (.*)", mood_input.lower())
            
            if match:
                content = match.group(1) or match.group(2)
                print(f"Reggy++: Why are you {content}?")
            else:
                if(a):
                    a=0
                    b=1
                    print("Reggy++: Man That is interesting, we should talk about it more or if you are busy just type quit!!")
                else:
                    b=0
                    a=1
                    print("Reggy++: Sounds worth the time, tell me in detail or if you are busy just type quit!!") 
                


# this runs the chatbot when executed from terminal
if __name__ == "__main__":
    main()
