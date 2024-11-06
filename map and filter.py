# بسم الله الرحمن الرحيم


f = lambda x: x ** 2
nums = [1, 2, 3, 4, 5]

# Question: how can I apply the function "f" on every element in "nums?
# --> Can use a for loop:
for i in nums:
    print(f(i), end=', ')
print("\n")

# But that takes a while, and requires a lot of code!
# Thus we use the "map" function:
x = map(f, nums)  # --> "map(function, iterable entry)"  --> passes the input arguments to be mapped "nums" into the mapping function "f"
# "f" maps input arguments to the output space (f = mapping)
# NOTE: "map()" is a generator method. Therefore it needs to be casted into a list!

for n in x:  # "x" now mapped each input in "nums" to the output space using the function "f"
    print(n, end=', ')
print("\n")

# OR:
y = list(map(f, nums))
print(f"y = {y} \n")


# Another eg:
def splice(string):
    if len(string) % 2 == 0:
        return "Even"
    else:
        return "Odd"


names = ["Omar", "Sultan", "Salim"]
check = list(map(splice, names))
print(check, "\n")


###################################################
# Using "filter" function: (used to testify True or False statements)
def even_check(num):
    return num % 2 == 0  # Filter will only return the elements that result in: True  --> "filters" out False entries


# "filter" will check the arguments against the conditions in your defined function
# If the passed inputs/arguments satisfy the conditions (True), it will return those arguments/inputs, else it wont return anything
nums = [1, 2, 3, 4, 5]
fil = list(filter(even_check, nums))
print(fil, "\n")

###################################################
# Lambda expressions:
# Normally, you wouldn't assign lambda functions to variables (you don't name them)
lam_map = list(map(lambda k: k**2, nums))  # map(lambda function, inputs/args)
print(lam_map, "\n")

# Similarly:
lam_fil = list(filter(lambda k: k % 2 == 0, nums))
print(lam_fil)
