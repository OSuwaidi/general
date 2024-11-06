# بسم الله الرحمن الرحيم

# *args = arguments
# **kwargs == keyword arguments

# If we don't define/assign a value to "c", it will be set automatically to 0, else it will be replaced by whatever we put in
def myfunc1(a, b, c=0):  # Here (a, b, c) are called positional arguments
    print(sum((a, b, c)) * 0.5, "\n")


myfunc1(20, 60, 20)


# Question: what if we wanted to work with MULTIPLE (a lot) of parameters? We can add many positional arguments (non-adaptive)
# OR:
# We can use "*args", which allows us to add an arbitrary number of arguments!
# eg:
def myfunc2(*args):  # Now this function takes as many arguments as you give it!
    print(args)  # IMPORTANT: the function takes the arguments as tuples!!!
    print(args[-1])  # RECALL: tuples can be indexed just like lists using indexing!!!
    print(sum(args) * 0.5, "\n")


myfunc2(40, 60)
myfunc2(40, 60, 100)
myfunc2(40, 60, 100, 200, 300)


def arg(*args):
    for x in args:
        print(x, end=" ")


arg([40, 50], 60, 100)  # Can even pass in a list as one of the arguments!
print("\n\n")


# **kwargs is used to build a dictionary of key values: (needs keyword arguments. eg: key='text' or key=#)
def dic(**kwargs):
    print(kwargs)  # kwargs is treated as a dictionary
    if 'fruit' in kwargs:  # This ONLY searches in the keys of a dictionary, not its values. "fruit" here is the key
        print(f"The fruit is {kwargs['fruit']} \n")  # "kwargs['fruit']" is the key VALUE
    elif 'bank' in kwargs:
        print(f"I have {kwargs['bank']} in my bank")
    else:
        print("No fruit :(", "\n")


dic(fruit='apple')  # 'fruit' is the keyword, and 'apple' is the value
dic(bird='eagle')  # 'bird' is the keyword and 'eagle' is the value
dic(bird='eagle', bank=100000, fish='whale')  # 'bank' is the keyword and 10000 is the value
print("\n")


# Combine the two:
def wow(*args, **kwargs):  # Order matters here!
    print(f"Your order is {kwargs['food']} for {args[2]} dirhams")


wow(10, 20, 70, fruit='orange', food='salmon', uni='MBZUAI')
