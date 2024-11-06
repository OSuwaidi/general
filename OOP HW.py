# بسم الله الرحمن الرحيم

# 1.) Fill in the "Line" class methods to accept coordinates as a pair of tuples and return the slope and distance of the line.
from math import sqrt


class Line:  # If we were to do this using normal functions, we would have multiple sub-functions enclosed by one large function, and we would have to call EACH AT A TIME!
    def __init__(self, pt1, pt2):  # pt = (x, y)
        self.pt1 = pt1
        self.pt2 = pt2

    def distance(self):
        x1, y1 = self.pt1  # Tuple unpacking!
        x2, y2 = self.pt2
        return sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def slope(self):
        x1, y1 = self.pt1
        x2, y2 = self.pt2
        m = (y2 - y1)/(x2 - x1)
        return m


coord1 = (3, 2)
coord2 = (8, 10)
lin = Line(coord1, coord2)
print(lin.distance())
print(lin.slope(), "\n")


# 2.) Construct a Cylinder class that accepts height and radius as arguments, and returns the surface area and the volume of that cylinder
class Cylinder:
    pi = 3.14

    def __init__(self, height, radius):
        self.height = height
        self.radius = radius

    def sa(self):
        return 2 * (self.pi * self.radius**2) + (2 * self.pi * self.radius * self.height)  # SA = 2 circles: top and bottom, then the curved side of the cylinder

    def vol(self):
        return self.pi * self.radius**2 * self.height


C = Cylinder(2, 3)
print(C.sa())
print(C.vol(), "\n\n")


# 3.) Create a bank account class that has 2 attributes: "owner" and "balance", and you should be able to PRINT the class object. With 2 methods: "deposit" and "withdraw", and withdrawals should not exceed the available balance.
class Account:
    def __init__(self, owner, balance):
        self.owner = owner
        self.balance = balance

    def __str__(self):
        return f"This is {self.owner}'s account, with a balance of ${self.balance}"  # THIS MUST BE "return", CANNOT USE "print"!!!

    def deposit(self, amount):
        self.balance += amount
        print(f"Deposited ${amount}")

    def withdraw(self, amount):
        if amount > self.balance:
            print("Insufficient funds!")
        else:
            self.balance -= amount
            print(f"Withdrawn ${amount}")


acct1 = Account("Omar", 7000)

print(acct1, "\n")  # Prints the string of the class object using "__str__" constructor

print(acct1.owner)
print(acct1.balance, "\n")

# We didn't use "print" here because our methods had "print" instead of "return"
acct1.deposit(3000)
acct1.withdraw(1000)
acct1.withdraw(10000)


#############################################
# You can use functions defined from other scripts by saying: from (your script file) import (the function you want to use)
# Eg:
# from Search import Linear

# You can also import entire scripts if you want by: from (the folder that contains the script) import (script name). OR: import numpy*


#############################################
# Question: what is "__name__"?
# __name__ gets assigned a "string" variable
# if __name__ == '__main__':  (running the script directly)
#    print("This file is being run directly")
# else:
#    print("This file is being imported")

# Say we have 2 script files, one called A, and another called B.
# If we run "A" directly with that statement, it will print: "This file is being run directly"
# However, if we call a function from "A" in script "B"; i.e if we import function from script "A" to script "B" as such:
# (In "B" script)
# import A      OR:     from A import fun_A()
# Then it will print "This file is being imported"

# Eg:
# If we had this following code in numpy package:
# if __name__ == '__main__':
#    print("numpy is being run directly")
# else:
#    print("numpy is being imported")

# If we imported numpy and ran it in our own code, it will print "numpy is being imported"
