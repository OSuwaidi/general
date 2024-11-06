# بسم الله الرحمن الرحيم
"""
Calling methods in different methods within the same class, using: "self.method()" in the other methods
Eg:
"""


class Villa:
    # Class object attributes:
    balance = 5 * 10**6  # Amount of money I have
    inflation = 1.10
    tax = 0.05

    def __init__(self, area, year):  # Recall: arguments inserted in the "__init__" constructor are also class arguments. Thus, we have to pass these arguments into the class when creating an instance of that class
        self.area = area
        self.year = year

    def cost(self):
        if self.year >= 2025:
            Villa.inflation = 1.20
        return 10**4 * self.area * Villa.inflation * (1 + Villa.tax)  # *** Has to be "return" NOT "print" to be able to call it as "self.cost()" when using it somewhere else ***

    def marriage(self, expenses):
        if Villa.balance - self.cost() >= expenses:  # Called a method within another method by using: "self.method()"
            print("Can get married!")
        else:
            print("Not enough money for marriage.")


house = Villa(350, 2021)
print(house.cost())
house.marriage(30000)
