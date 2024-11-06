# بسم الله الرحمن الرحيم
# To handle exceptions, and to call code when an exception occurs, you can use a try/except statement.
# The try block contains code that might throw an exception.
# If that exception occurs, the code in the try block stops being executed, and the code in the except block is run.
# If no error occurs, the code in the except block doesn't run.

try:
    num1 = 7
    num2 = 0
    print(num1 / num2)
    print("Done calculation")
except ZeroDivisionError:
    print("An error occurred")
    print("Cannot divide by zero division \n")

try:
    variable = 10
    print(variable + "hello")
    print(variable / 2)
except ZeroDivisionError:
    print("Divided by zero")
except (ValueError, TypeError):
    print("Incorrect type/value \n")

# "except:" without any arguments will occur/run if *ANY* error is raised
try:
    x = 0
    y = 0
    z = x / y
except:
    print('Try adjusting parameters')
