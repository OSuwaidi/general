# بسم الله الرحمن الرحيم
"""
* The libraries you call from python, such as "from numpy" or "from torch" are an all examples of classes. Where "numpy" and "torch" are classes that contain different methods and attributes
* When you access these classes; calling their objects/instances followed by a ".", you are using their attributes, and when calling their objects/instances followed by a ".()" you are using their methods
* As an example: "numpy.exp()" (object.method), (where "numpy" = object, "exp" = method)
                 "torch.data" (object.attribute), (where "torch" = object, "data" = attribute)
"""


class Circle:  # Creating a class
    Radius = 4  # Attributes/Properties
    Pi = 3.14  # Attributes/Properties
    Area = (Pi * Radius ** 2)  # Method (not "really" a method, method needs to be a function "def" inside of a class)


# *** Note ***: In reality, the "Circle" class above is a 'class oriented class', instead of an 'object oriented class', i.e: we can access the class's methods and attributes by calling the class DIRECTLY:
# Circle.Radius
# Circle.Area

# Instance/Object of a class ("object1" is an object of the class "Circle"):
object1 = Circle()  # Define an object "object1" to be called, that contains the attributes/properties and methods which are defined in the class "Circle" above

Area = object1.Radius**2 * object1.Pi
print(f"Area = {Area}")
print("Area_method =", object1.Area, "\n")  # Can just call the method from object "object1"  --> This is an attribute in reality

# Assigning new values for attributes/properties, Radius and Pi:
object1.Radius = 2
object1.Pi = 3.1421
Area_new = object1.Radius ** 2 * object1.Pi
print(f"Area_new = {Area_new}")
print("Area_method=", object1.Area, "\n\n")  # NOTICE: Reassigning the values for the attributes DID NOT ACTUALLY CHANGE THE ATTRIBUTES WITHIN THE CLASS!!!

# Therefore, calling the method "object1.Area" still prints out the old area, which has the previous attributes as defined in the class


# To make a method that can reassign attributes within a class:
class SomeCircle:
    pi = 3.14  # Class object attribute
    # *** Here ***: you add what is shared/common in the ENTIRE class (eg: if you have 5 circles with different radii, you add "pi" in "Class object attributes" because ALL circles have same "pi" value, but you add "r" in "init" attributes, because not all the circles share the same radius, therefore need different "r's" for different circles
    # Note: adding "__init__" under a class makes the class non-static

    def __init__(self, r=2):  # The "__init__" initializes the values of the object's attributes each time you call an object/instance of that class
        self.r = r
        # *** Here ***: you add what is required as an argument for every object of the class

    # After we assigned the attributes in the class (above), we now define the methods that uses/calls these attributes (below)
    def calc(self):  # Adding "self" inside the method passes-down/gives-access to all the defined attributes. Now it is an instance/object method, as opposed to a static method
        A = self.pi * self.r ** 2
        perimeter = 2 * SomeCircle.pi * self.r  # Better to use "SomeCircle.pi", since "pi" is a class object attribute, not an "instance attribute"
        diameter = 2 * self.r
        print(f"Area = {A} \nPerimeter = {perimeter} \nDiameter = {diameter} \n")


c1 = SomeCircle(1)  # Create object/instance "c1" under class "SomeCircle" with "r"=1 instead of the default "r"=2
c1.calc()


# Executing the same method as above but making it static this time:
class StaticCircle:  # A static method is bound to a class, rather than the objects for that class. Thus, it can be called WITHOUT using an object!
    @staticmethod
    def calc(r=2):  # ***Now since we didn't use any class attributes in this class (no "self." attributes under "init")***, "calc()", is now a static method!
        A = 3.14 * r ** 2  # The static method is only dependent/bound to the class, NOT objects of that class
        perimeter = 2 * 3.14 * r
        diameter = 2 * r
        print(f"Area_static = {A} \nPerimeter_static = {perimeter} \nDiameter_static = {diameter} \n\n")


StaticCircle.calc(1)  # We can call static methods from the class DIRECTLY, without the need to use instances/objects (or even the need to create them)
                      # Notice that the class itself takes the method's argument


# Understanding "super":
class Animal:  # Parent "Animal" class requires 2 arguments
    def __init__(self, speed, is_mammal):
        self.speed = speed
        self.is_mammal = is_mammal


class Cat(Animal):  # Child "Cat" class requires 1 argument
    def __init__(self, is_hungry):
        super().__init__(7, True)  # Since parent class "Animal" takes 2 arguments, so does child class "Cat"
        self.is_hungry = is_hungry


barry = Cat(True)
print(f"speed: {barry.speed}")
print(f"Is a mammal?: {barry.is_mammal}")
print(f"Feed the cat?: {barry.is_hungry} \n\n")


# To explain the "__init__" constructor:
class Test:
    grades = []  # Class attribute


test1 = Test()  # First object/instance
test2 = Test()  # Second object/instance
test1.grades.append(98)
print(f"test1 = {test1.grades} \ntest2 = {test2.grades} \n")  # Even though we never assigned any value for object "test2", it still contained the value of object "test1"


# Resetting the attributes (initializing) fixes this problem:
class Test:
    def __init__(self):
        self.grades = []


test1 = Test()  # First object/instance
test2 = Test()  # Second object/instance
test1.grades.append(98)
print(f"test1 = {test1.grades} \ntest2 = {test2.grades} \n\n")


# An example of a class with multiple (4) instance/object static methods (without using "__init__"):
class AreaFinder:  # These shapes don't share anything in common, thus there is no need to create the "__init__" constructor and add attributes (common properties)
    def square(self, side):
        print(f"A_square = {side ** 2}")

    def rectangle(self, length, width):
        print(f"A_rectangle = {length * width}")

    def triangle(self, base, height):
        print(f"A_triangle = {0.5 * base * height}")

    def circle(self, radius):
        print(f"A_circle = {3.14 * radius ** 2} \n\n")


area = AreaFinder()
area.square(3)
area.rectangle(4, 5)
area.triangle(4, 3)
area.circle(2)


# You can also have input arguments for your class AND your object:
class Human:
    def __init__(self, name1):  # The "__init__" arguments are to be called from the class itself!!!
        self.name1 = name1  # Here "name1" is the attribute assigned to class "Human"
        print(f"Main character is {name1}!")

    def kill(self, name2):
        print(f"{self.name1} killed {name2}!!!")

    def love(self, name2):
        print(f"{self.name1} loves {name2} <3")

    def friends(self, name2):
        print(f"{self.name1} is friends with {name2}.")


act = Human("Omar")  # What will be shared with the ENTIRE class
act.kill("Sultan")
act.love("Sultan")
act.friends("Sultan")
print("\n")


# Example of inheritance in classes:
class Family:
    def __init__(self, last_name):
        self.lastname = last_name


class Father(Family):
    def dad(self):
        print("Ali")


class Mother(Family):
    def mom(self):
        print("Zubaida")


class Son(Mother, Father):
    def name(self):
        print("Omar")


Me = Son('AlSuwaidi')  # Even though class "Son" takes no input arguments (no "__init__"), the Parent class "Family" does take one, so in turn, class "Son" also requires an input argument!
Me.name()
Me.mom()
Me.dad()
print(Me.lastname)  # Since this is an attribute, we don't call it using "()"
