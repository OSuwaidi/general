# بسم الله الرحمن الرحيم
# Functions Exam:

# 1.) LESSER OF TWO EVENS: Write a function that returns the lesser of two given numbers if both numbers are even, but returns the greater if one or both numbers are odd
def lesser(a, b):
    if a % 2 == 0 and b % 2 == 0:
        return min(a, b)  # This is treated as a tuple
    else:
        return max(a, b)


print(lesser(2, 4))
print(lesser(2, 5), "\n")


# 2.) ANIMAL CRACKERS: Write a function takes a two-word string and returns True if both words begin with same letter
def animals(text):
    words = text.split()  # Returns a LIST of strings which were separated by a space " "
    if words[0][0] == words[1][0]:
        return True
    else:
        return False


print(animals("Fishy Fish"))
print(animals("Fishy Wish"), "\n")


# OR:
def anim_crack(text):
    word_list = text.split()
    return word_list[0][0] == word_list[1][0]


print(anim_crack("Fishy Fish"))
print(animals("Fishy Wish"), "\n")


# 3.) MAKES TWENTY: Given two integers, return True if the sum of the integers is 20 or if one of the integers is 20. If not, return False
def is_20(a, b):
    if a == 20 or b == 20:
        return True
    elif a + b == 20:
        return True
    else:
        return False


print(is_20(20, 10))
print(is_20(12, 8))
print(is_20(2, 3), "\n")


# OR:
def twenty(a, b):
    add = a + b
    if a == 20 or b == 20 or add == 20:
        return True
    else:
        return False


print(twenty(20, 10))
print(twenty(12, 8))
print(twenty(2, 3), "\n")


# 4.) OLD MACDONALD: Write a function that capitalizes the first and fourth letters of a name
def mac(name):
    first = name[:3].capitalize()  # ".capitalize()" capitalizes the FIRST letter ONLY in a string
    second = name[3:].capitalize()
    return first + second


print(mac("macdonald"), "\n")


# 5.) MASTER YODA: Given a sentence, return a sentence with the words reversed
def rev(sentence):
    return " ".join(sentence.split()[::-1])  # Reverses the order in the list, then joins the strings in that list


print(rev('I am home'))
print(rev('We are ready'), "\n")


# 6.) ALMOST THERE: Given an integer n, return True if n is within 10 of either 100 or 200
def check(n):
    if abs(100 - n) <= 10:
        return True
    elif abs(200 - n) <= 10:
        return True
    else:
        return False


print(check(104))
print(check(150))
print(check(209), "\n")


# OR:
def hund(n):
    if abs(n - 100) <= 10 or abs(n - 200) <= 10:
        return True
    else:
        return False


print(hund(104))
print(hund(150))
print(hund(209), "\n")


# 7.) FIND 33: Given a list of ints, return True if the array contains a 3 next to a 3 somewhere.
def has_33(numbers):
    for i in range(1, len(numbers)):  # Recall: range goes from 1 and ends at len(numbers) NOT INCLUDING (excluding) len(numbers)
        if numbers[i - 1] == 3 and numbers[i - 1] == numbers[i]:  # We did it using [i-1], because if we used [i+1] we would eventually go out of list index range
            return True
    else:  # IMPORTANT! We took the "else:" statement OUT of the for loop, because if we didn't, it would've always returned "False" from the first iteration, and it would've terminated/stopped the function from further iterations because of it's "return False" statement (Try putting in for loop and see!)
        return False


print(has_33([1, 3, 3]))
print(has_33([1, 2, 1, 3]))
print(has_33([3, 1, 3]), "\n")


# OR:
def three(arr):
    for i in range(len(arr) - 1):  # (-1) so that we wouldn't get "list index out of range"
        if arr[i] == arr[i + 1] == 3:  # Can take multiple inequalities!
            return True
    else:
        return False


print(three([1, 3, 3]))
print(three([1, 2, 1, 3]))
print(three([3, 1, 3]), "\n")


# 8.) PAPER DOLL: Given a string, return a string where for every character in the original there are three characters
def triple(string):
    for c in string:
        print(c * 3, end="")
    print("")


triple('Hello')
triple('Mississippi')
print("\n")


# OR:
def trip(text):
    result = ''
    for char in text:
        result += char * 3
    return result


print(trip('Hello'))
print(trip('Mississippi'), "\n")


# 9.) BLACKJACK: Given three integers between 1 and 11, if their sum is less than or equal to 21, return their sum. If their sum exceeds 21 and there's an eleven, reduce the total sum by 10. Finally, if the sum (even after adjustment) exceeds 21, return 'BUST'
def bjack(x, y, z):
    tot = x + y + z  # Or: tot = sum((x, y, z))
    if tot <= 21:
        return tot
    elif tot > 21 and (x == 11 or y == 11 or z == 11):
        if tot - 10 > 21:
            return "BUST"
        else:
            return tot - 10
    else:
        return "BUST"


print(bjack(5, 6, 7))
print(bjack(9, 9, 9))
print(bjack(3, 2, 11))
print(bjack(11, 11, 10), "\n")


# OR:
def jack(x, y, z):
    tot = sum((x, y, z))
    if tot <= 21:
        return tot
    elif 11 in (x, y, z) and (tot - 10) <= 21:  # We reduced by 10 here, because tot would have an 11, therefore we deduct 10 and then check with 21
        return tot - 10
    else:
        return "BUST"


print(jack(5, 6, 7))
print(jack(9, 9, 9))
print(jack(3, 2, 11))
print(jack(11, 11, 10), "\n")


# 10.) SUMMER OF '69: Return the sum of the numbers in the array, except ignore sections of numbers starting with a 6 and extending to the next 9 (every 6 will be followed by at least one 9). Return 0 for no numbers.
def summer_69(array):
    summer = 0
    start = 0
    for i in range(len(array)):
        if array[i] != 6:
            summer += array[i]
        else:  # Else if array[i] was == 6, then break the for loop
            start = i
            break
    for i in range(len(array[start:])):
        i += start
        if array[i] == 9:
            for n in array[i + 1:]:  # We used [i + 1] to skip the value of 9
                summer += n
            break  # Only need to run this for loop once, AKA: Stop this operation after the first 9 you see. Because if we didn't break, it would apply this operation again for other 9's it stops after our first 9!
    return summer


print(summer_69([1, 3, 5]))
print(summer_69([4, 5, 6, 7, 8, 9]))
print(summer_69([2, 1, 6, 9, 11]))
print(summer_69([9, 2, 6, 3, 9, 10, 9, 2]), "\n")  # If we didn't break 2nd for loop, it would add this "2" twice, because it would've added the [i+1] term after the 2nd 9 too!


# CHALLENGING PROBLEMS AHEAD!
# 10.) SPY GAME: Write a function that takes in a list of integers and returns True if it contains 007 in order
def bond(arr):
    f = 0
    code = [0, 0, 7]  # List we want to match
    for i in range(len(arr)):
        if arr[i:(i+3)] == code:  # Recall: last element in arr[a:b] is not included (b not included)
            f = 1  # Can't say "return True" here because if we do, it will stop/terminate the function at the very first iteration
    if f == 1:  # Therefore we created a flag
        return True
    else:  # If we didn't specify this, it would've returned "None"
        return False


print(bond([1, 2, 4, 0, 0, 7, 5]))
print(bond([1, 0, 2, 4, 0, 5, 7]))
print(bond([1, 7, 2, 0, 4, 5, 0]), "\n")


# OR:
def bond2(arr):
    for i in range(len(arr) - 1):  # We used [len(arr)-1] to avoid "list index out of range". Read 7.) above for more info
        if arr[i] == arr[i + 1] == 0 and arr[i + 2] == 7:  # IMPORTANT! "if" statement can take more than one argument/comparison operator!!!
            return True
    else:  # Took it outside the for loop, so that it wouldn't print "False" after every iteration
        return False


print(bond2([1, 2, 4, 0, 0, 7, 5]))
print(bond2([1, 0, 2, 4, 0, 5, 7]))
print(bond2([1, 7, 2, 0, 4, 5, 0]), "\n")


# 11.) COUNT PRIMES: Write a function that returns the number of prime numbers that exist up to and including a given number
def prime_count(z):
    prime_N = 0
    primes = []
    for x in range(2, z + 1):  # We started from 2, because 0 and 1 are definitely not prime numbers, thus no need to check for them. Also we ended with (z+1) because we want to include the value of "z"
        test = []  # Note: range(b, a) where b >= a, returns an empty list!
        for t in range(2, x):  # We started from 2, because every prime number is divisible by 0 and 1, and ended at "x" (x-1 really), because every prime number is also a factor of itself
            test.append(x % t)  # Check if "x" is divisible by every number from 2 to (x-1). If it IS divisible--> remainder(%) == 0--> (0 not in test = False)--> x is NOT prime
        if 0 not in test:
            primes.append(x)
            prime_N += 1
    return prime_N, primes


print(prime_count(100))
print(prime_count(25))
print(prime_count(2), "\n")  # test = [] (empty list), and technically, 0 is not in test. Therefore "prime_N += 1" gets executed! الحمدلله!


# 12.) PRINT BIG: Write a function that takes in a single letter, and returns a 5x5 representation of that letter
def big(letter):
    patterns = {1: '  *  ', 2: ' * * ', 3: '*   *', 4: '*****', 5: '**** ', 6: '   * ', 7: ' *   ', 8: '*   * ', 9: '*    '}
    alphabet = {'A': [1,2,4,3,3], 'B': [5,3,5,3,5], 'C': [4,9,9,9,4], 'D': [5,3,3,3,5], 'E': [4,9,4,9,4]}
    for pattern in alphabet[letter.upper()]:
        print(patterns[pattern])
    print("\n")


big('b')

# 12 Explanation:
# When you define a dictionary, and then run a for loop as such:
dic = {'one': [1, 2, 3], 'two': [2, 3, 4], 'three': [3, 4, 5]}
for x in dic:
    print(x)  # It prints the "KEYS" in the dictionary not the "values"!

print("\n")

# You can also call a dictionary like you call a list, but instead of taking an index, it takes the "key":
print(dic['one'])
print(dic['three'], "\n")

# Since we have a list as  "values" for our "keys":
for x in dic['one']:
    print(x)  # Will print the values in the LIST of "key" 'one'


# 13.) Define a function that returns a string with even letters capitalized and odd letters not
def myfunc(text):
    i = 0
    f = len(text)
    string = ''
    while i < f:
        if i % 2 == 0:
            string += text[i].capitalize()
        else:  # "else" here will take every number that is not even: (1, 3, 5, ...)
            string += text[i]
        i += 1
    print(string)


myfunc("hello")
