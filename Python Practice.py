# بسم الله الرحمن الرحيم

mylist = [x for x in range(0, 11) if x % 2 == 0]
print(mylist)

# Nested in list comprehension:
nested = [x * y for x in (1, 10, -1) for y in (1, 2, 3)]
print(nested)
# This is the same as:
# nested = []
# for x in (1, 10, -2):
    # for y in (1, 2, 3):
    # nested.append(x*y)


# Statements TEST:

# 1.) Use for, .split(), and if to create a Statement that will print out words that start with 's':
st = 'Print only the words that start with s in this sentence'
# Note: "st.split()" without any arguments inserted (by default) will split the sentence whenever it sees a space " "
for x in st.split():  # Note: "st.split()" is a list that is iterable like any other normal list!
    if x[0] == "s":
        print(x, end=" ")
print("\n")

# 2.) Use range() to print all the even numbers from 0 to 10:
for y in range(11):
    if y % 2 == 0:
        print(y, end=" ")
print("\n")
# OR:
list(range(0, 11, 2))

# 3.) Use a List Comprehension to create a list of all numbers between 1 and 50 that are divisible by 3.
a = [n for n in range(1, 51) if n % 3 == 0]
print(a, "\n")

# 4.) Go through the string below and if the length of a word is even print "even!"
st = 'Print every word in this sentence that has an even number of letters'

for k in st.split():
    if len(k) % 2 == 0:
        print(k, end=" ")
print("\n")

#*** 5.) Write a program that prints the integers from 1 to 100. But for multiples of three print "Fizz" instead of the number, and for the multiples of five print "Buzz". For numbers which are multiples of both three and five print "FizzBuzz".
for i in range(1, 101):
    if i % 15 == 0:  # If this statement was used as an "elif" statement, it would have never been executed! Because (i%15==0) will never be true unless (i%3==0) was also true. Thus if the statement (i%3==0) preceded the (i%15==0) statement, (i%3==0) would have ALWAYS been executed first (before i%15) and it would never reach the (i%15==0) statement!
        print("FizzBuzz", end=" ")  # Can also use: if num % 3 == 0 and num % 5 == 0:
    elif i % 3 == 0:
        print("Fizz", end=" ")
    elif i % 5 == 0:
        print("Buzz", end=" ")
    else:
        print(i, end=" ")
print("\n")

# 6.) Use List Comprehension to create a list of the first letters of every word in the string below:
st = 'Create a list of the first letters of every word in this string'
h = st.split()  # "h" is now a list of all the words in "st" that were separated by a spacebar " "
letters = [L[0] for L in h]
print(letters, "\n")


######################################################################################################################


def even_check(x):
    return x % 2 == 0  # This is saying return "True" if (x%2==0), otherwise return "False" if not


print(even_check(2), "\n")


def even_list(lissst):
    for number in lissst:
        if number % 2 == 0:
            return True  # NOTE: once you your function reaches "return" it breaks off the function and ends it!
    return False  # This "return" is outside the "if" statement, therefore it wont be terminated by the above "return"


a = [1, 2, 5]
print(even_list(a), "\n")

# Q.) Find out the employer of the year based on work hours, and print his name + amount of work hours:
work_hours = [("Sultan", 500), ("Omar", 900), ("Salim", 200)]


def EOY(work):
    max_hours = 100
    winner = ""
    for employee, hours in work:
        if hours > max_hours:
            max_hours = hours
            winner = employee
    print(f"Employer of the year is {winner} with {max_hours} hours!")


x = EOY(work_hours)  # Statement gets printed here
print(x, "\n")  # Notice that "x" is type "None", because I can't store/assign "print" statement in a variable

# To fix the assignment issue, I can just use "return" instead of "print" in my function

def EOY(work):
    max_hours = 100
    winner = ""
    for employee, hours in work:
        if hours > max_hours:
            max_hours = hours
            winner = employee
    x = f"Employer of the year is {winner} with {max_hours} hours!"
    print(x)
    return x


EOY(work_hours)
print(x)  # "x" data is only stored *inside* (within) the function
print(EOY(work_hours))  # prints twice, because one print is in the function, and the other is printing the "return" statement
