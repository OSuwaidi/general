# بسم الله الرحمن الرحيم

# 1.) Write a function that computes the volume of a sphere given its radius.
from math import pi

radii = [2, 3, 5]
volume = list(map(lambda r: 4 / 3 * pi * r ** 3, radii))
print(f"List of volumes = {volume}")


# OR:
def vol(r):
    return 4 / 3 * pi * r ** 3


print(f"Sphere volume = {vol(2)} \n")


# 2.) Write a function that checks whether a number is in a given range (inclusive of high and low)
def check1(x, low, high):
    if x >= low and x <= high:
        return f"{x} is in the range between {low} and {high}"
    else:
        return False


# Can make this function simpler:
def check2(x, low, high):
    if low <= x <= high:  # Chained comparison
        return f"{x} is in the range between {low} and {high}"
    else:
        return False


print(check1(5, 2, 7))
print(check2(5, 2, 7), "\n")


# 3.) Write a Python function that accepts a string and calculates the number of upper case letters and lower case letters.
def up_low(string):
    Upper = 0
    Lower = 0
    for c in string:
        if c.isupper():  # "isupper()" checks whether the character called is upper case or not. If it is upper, it returns True (and therefore it executes the "if" statement), else it returns False
            Upper += 1
        elif c.islower():  # We didn't use "else:" only, because it would count the spaces, commas and punctuations as characters!
            Lower += 1
    print(f"# of Upper case characters = {Upper}")
    print(f"# of Lower case characters = {Lower} \n")


s = 'Hello Mr. Rogers, how are you this fine Tuesday?'
up_low(s)


# 4.) Write a Python function that takes a list and returns a new list with unique elements of the first list.
def unique(lis):
    uni = set()  # Recall: whatever you add in sets, it ONLY shows/takes unique elements
    for x in lis:
        uni.add(x)
    uni = list(uni)
    return uni


print(unique([1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 5]), "\n")


# OR:
def unique2(lis):
    return list(set(lis))


print(unique2([1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 5]), "\n")


# 5.) Write a Python function to multiply all the numbers in a list.
def mult(arr):
    prod = 1
    for y in arr:
        prod *= y
    return prod


print(mult([1, 2, 3, -4]), "\n")


# 6.) Write a Python function that checks whether a word or phrase is palindrome or not.
# A "palindrome" word/phrase means that the word/phrase is readable from both directions: left to right and right to left (eg: kayak)
def palindrome(text):
    if text.replace(" ", "") == text.replace(" ", "")[::-1]:  # Replace spaces " " in text with no space "". Done to compare characters ONLY, not spaces
        return True
    else:
        return False


print(palindrome('helleh'))
print(palindrome('nurses run'), "\n")

# 7.) HARD: Write a Python function to check whether a string is pangram or not.
# A "pangram" is a unique word/sentence in which every letter of the alphabet is used at least once
import string


def pangram(text):
    alphabet = []  # We could write every letter of the alphabet in this list to compare against our string, but that would be too lengthy
    alphabet = string.ascii_lowercase
    checker = []
    for letter in alphabet:
        if letter in text.lower():  # We lowered every character in "text", because we are using lowercase letters to compare against letters in our text
            checker.append(True)  # This will check every letter in the alphabet against letters in our text; if they all exist (all True), then we return True
        else:
            checker.append(False)
    return False not in checker  # If "False" is NOT in checker, it would return "True", else it would return "False"


print(pangram("The quick brown fox jumps over the lazy dog"))
print(pangram("My name is Omar AlSuwaidi"))
