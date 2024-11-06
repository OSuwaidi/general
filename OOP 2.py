# بسم الله الرحمن الرحيم

# "Class" is kind of the "object" itself
# Under the Class, "Class object attributes" come first, then come the "self attributes" (shared across methods) which are under the "init" constructor
# "Class object attributes" are attributes that are shared across ALL |objects/instances| of a class. "Self attributes" are attributes that are shared across ALL |METHODS| within an object/instance of a class
# Then you define methods to be called and act on the attributes
# Eg:
class Bank:
    def __init__(self, money):  # Note: everytime you want to pass down an argument from one method to another, you HAVE to use "self"
        self.money = money  # Now argument became an attribute

    def withdraw(self, amount):
        self.money -= amount  # Notice we use (self.___) when we want to use it/pass it to other methods below

    def deposit(self, amount):
        self.money += amount

    def balance(self):  # "self" connects methods together within a class
        return self.money


# Lets create an instance/object of 'BankAccount' class, with the initial amount of money as $5000:
BankAccount = Bank(5000)
print(BankAccount.balance(), "\n")

# Lets deposit the salary:
BankAccount.deposit(500)

# Lets withdraw some money:
BankAccount.withdraw(200)

# What's the balance left?
print(BankAccount.balance(), "\n")


##################################################################
class Dog:
    kind = "Mammal"  # "Class object attributes" (COA): are attributes that are shared across all objects/instances of class "Dog"

    # Note that the "Class object attributes" (kind="Mammal") SHOULD be called using classes---> class.COA---> Dog.kind
    def __init__(self, breed, name, color):  # These are attributes to be shared with all methods, and assigned to each object of a class based on arguments passed!
        self.type = breed  # Assign the argument "breed" to the ATTRIBUTE "self.type"
        self.name = name
        self.color = color

    def bark(self, times):  # Methods acts on attributes of a class
        sound = ("WOOF! I am " + self.name + ". ") * times  # Note that "self.argument" will only be used when I'm calling an attribute (from init), not when calling arguments!!!
        return sound


dog1 = Dog("Huskie", "Bobby", "Brown")
dog2 = Dog("Pug", "Pup", "Gray")

print(Dog.kind, dog1.type, dog1.name, dog1.color)  # Notice that class object attributes were called using the class itself
print(Dog.kind, dog2.type, dog2.name, dog2.color)
print(dog1.bark(2))  # Notice that methods have "()", while attributes DON'T!
print(dog2.bark(3), "\n\n")


##################################################################
class Circle:
    pi = 3.14  # Class object attribute

    def __init__(self, radius=1):  # If no radius is passed in as an argument, it will by default use a radius of 1, unless specified otherwise
        self.r = radius
        self.area = self.pi * radius ** 2  # If we didn't add "self" before "area", we would have created a variable called "area", but it wouldn't be callable!

    def get_circumference(self):  # Why not assign "r" here, with the method itself? Because "radius" is an argument that will be used for ALL circles, not just for circumference, or area, or volume methods. Thus if you the "r" argument for each method (area, volume, etc.), then you would need to pass "r" as an argument into EACH method. Therefore, only assign it once under "init" (class argument) to be used everywhere!
        return 2 * self.pi * self.r  # --> Better to use "Circle.pi"

    def get_volume(self):
        return 4 / 3 * Circle.pi * self.r ** 3  # IMPORTANT: We called "pi" using the class it self, not "self"  --> "Circle.pi" instead of "self.pi", and that is to emphasize that the variable "pi" is a class object attribute!!!


# We could have defined "get_circumference" and "get_volume" under the "init" constructor, since they really don't need to be functions/methods as don't take any arguments, nor are they complex.
    # def __init__(self, radius=1):
        # self.get_circumference = 2 * Circle.pi * self.r
        # self.get_volume = 4 / 3 * Circle.pi * self.r ** 3
# my_circle.get_circumference
# my_circle.get_volume

my_circle = Circle()  # Even though we don't pass any arguments, it actually requires one! But it runs because "radius=1" by default!
print(f"Circumference = {my_circle.get_circumference()} \n")

my_circle = Circle(3)
print(f"Area = {my_circle.area}")  # Did not use "()" at the end, because "area" here is an attribute not a method
print(f"Circumference = {my_circle.get_circumference()}")
print(f"Volume = {my_circle.get_volume()} \n\n")


##################################################################
class Animal:  # Parent/Super class
    def __init__(self):
        print("ANIMAL CREATED")

    def what(self):  # PyCharm thinks that "what" and "eat" methods are static, because they never used any attributes from the "__init__" constructor
        print("I am an animal")

    def eat(self):
        print("I am eating")


class Cat(Animal):  # Now child/sub class "Cat" has access to all the attributes and methods of the parent/super class "Animal"
    def __init__(self, name):
        super().__init__()  # We need to explicitly call the super class initializer, so that you get access to what's defined under "__init__" in the parent/super class --> Only required when sub/child class has its own "init"
        print("It's a cat!")
        self.name = name


my_cat = Cat('Snow')  # Notice that the statements under "init" constructor of both classes were printed just by assigning object to class!
my_cat.what()  # Using methods from Parent class "Animal"
my_cat.eat()
print(my_cat.name)
print("\n")


# Question: what if I wanted to reassign/change one of the Parent class attributes, like the "who" method?
class Cat(Animal):
    def __init__(self):
        Animal.__init__(self)
        print("It's a cat!")

    def what(self):  # Overwrite the method using the same method name
        print("I'm Batman")

    def meow(self):  # Create new methods if we want
        print("Meow!")


my_cat = Cat()
my_cat.what()
my_cat.meow()
print("\n")


##################################################################
class Owl:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return "Kookoo, I am " + self.name


class Parrot:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return "Prrrt, I am " + self.name


alex = Owl('Alex')
mike = Parrot('Mike')
print(alex.speak())
print(mike.speak(), "\n")

for pet in [alex, mike]:  # List of objects of a class
    print(type(pet))  # This is an example of Polymorphism!
    print(pet.speak())  # Even though they share the same method name "speak", they produce different outputs
print("\n")


def pet_speak(pet):
    print(pet.speak())


pet_speak(alex)
print("\n")


##################################################################
# "Abstract methods" are methods that are defined within a Parent/Super class. But that parent class is never assigned to any object/instance. Only there for inheritance purposes!
# Eg:
class Cuisine:  # ***NOTE: Even though sub classes: "Pizza and Burger" require no input arguments, their parent/super class "Cuisine" does, THEREFORE they both also require all the input arguments their super class takes/requires!!!
    def __init__(self, place):
        self.place = place


class Pizza(Cuisine):
    def meal(self):
        return f"Pizza from {self.place}"


class Burger(Cuisine):
    def meal(self):
        return f"Burger from {self.place}"


Lunch = Pizza("Italy")  # Note: Even though class "Pizza" has no input arguments, but its parent class "Cuisine" does; therefore it also requires that input argument!!!
Dinner = Burger("America")  # Same applies to "Burger"

print(Lunch.meal())
print(Dinner.meal(), "\n", "\n")


##################################################################
class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages


b = Book('Master AI', 'MBZUAI', 500)
print(b, "\n")  # Printing "book" doesn't show anything, only shows the place where this variable is stored in my memory


# To fix this, introduce Magic methods:
class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages

    def __str__(self):  # Now whenever a method calls the string representation of our class, it will return the following:
        return f"New book by {self.author} on how to {self.title}, which has {self.pages} pages"


b = Book('Master AI', 'MBZUAI', 500)
print(b, "\n")


# print(len(b))  # Gives an error


# Same magic method applies to length:
class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages

    def __str__(self):  # Now whenever a method calls the string representation of our class, it will return the following:
        return f"New book by {self.author} on how to {self.title}, which has {self.pages} pages"

    def __len__(self):  # Now when you take the length of your object class, the following will be returned:
        return self.pages


b = Book('Master AI', 'MBZUAI', 500)
print(len(b), "\n")


# If you want an action to occur when you delete a variable using "del":
class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages

    def __str__(self):  # Now whenever a method calls the string representation of our class, it will return the following:
        return f"New book by {self.author} on how to {self.title}, which has {self.pages} pages"

    def __len__(self):  # Now when you take the length of your object class, the following will be returned:
        return self.pages

    def __del__(self):
        print("You deleted that book!")


b = Book('Master AI', 'MBZUAI', 500)
del b
