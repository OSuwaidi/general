# بسم الله الرحمن الرحيم
# Recursion example (a function that calls itself):

def factorial(x):
    y = 1
    while x > 1:
        z = x * (x - 1)
        x -= 2
        y *= z
    return print(y)


factorial(7)


# Using recursion:
def recur(x):
    if x > 1:
        return x * recur(x - 1)
    else:
        return x


print(recur(5))
