# بسم الله الرحمن الرحيم

# Global variables are variables that are preassigned. That is, assigned outside the scope of a function (independent)
# Local variables are variables that are assigned WITHIN a function (dependent)
# Enclosed local variables are variables that are not assigned locally within the function itself, but assigned within the enclosing function
# Variable priority is: Local -> Enclosed -> Global (LEG)

# Example:
x = 50  # "x" here is a GLOBAL variable (Global assignment)


def test1():
    x = 40  # "x" here is a LOCAL variable to "test1", but is an ENCLOSED local variable (from the "enclosing scope") to "test2"

    def test2():  # closure (inner function)
        x = 30  # "x" here is a LOCAL variable to "test2"
        print(x)
    test2()


# Python looks at Local variables FIRST, then Enclosed Local, then Global variables:
test1()

print(x, "\n\n")  # Notice that reassigning variables locally DOES NOT AFFECT the global variables


#######################################################
# What if we wanted to change the global variable locally (inside a function)?
# Use "global" keyword:
x = 50


def func():  # DOES NOT TAKE "x" AS AN ARGUMENT!!!
    global x  # Very risky!
    x = 100
    print(x)


func()
print(x, "\n")  # Notice that it changed the "x" value Globally!

# A safer and a better way is to:
x = 50


def func2():
    x = 200
    return x


print(x)  # Global value of variable "x" (x = 50)
x = func2()  # Reassigning the value of the global "x" to now be the value of local "x" (x = 200)
print(x)  # After reassigning the value of "x"
