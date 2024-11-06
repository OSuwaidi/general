# بسم الله الرحمن الرحيم

def Linear(value, list):
    for i in range(len(list)):  # len(L1) = 8, thus "i" goes from 0 to 8. However RECALL that the last number isn't included!!! (goes from 0 to 8, excluding 8)
        if list[i] == value:  # Note: len(L1) is an integer, therefore it is not iterable
            print(f"Index = {i}")
            break  # Stops EVERYTHING below this line (as long as it's under the "for" statement) (This is saying that: if the above condition is met, break the "for" loop)
        elif i == len(L1) - 1:  # "i" was set to len(L1) - 1, because len(L1) = 8, and the last index in our list (L1) is 7. (When you set "i" = range(0, 8), it does not count the 8, but when you set "i" == len(L1), that's the same as saying "i" == 8)
            print("Value not found")  # "elif" will only occur if the "if" statement above it was not met

    print("This is linear search \n")  # Not under the "if" statement. Therefore "break" did not stop this from printing (It is also not under the "for" statement)


L1 = [1, 5, 9, 0, 10, 8, 7, 2]
L1.sort()
Linear(7, L1)
Linear(3, L1)


# Construct Binary Search and compare it to linear search. Compare execution speed of each search for the same list
L1 = [1, 5, 9, 0, 10, 8, 7, 2]
L1.sort()
print(L1)  # For binary search to work, the list needs to be arranged/sorted in ascending order


def Binary(value, list):
    holder = 0
    mid = (len(list)-1)//2
    while list[mid] != value:
        if list[mid] > value:
            list = list[0:mid]
            mid = (len(list) - 1) // 2
        elif list[mid] < value:
            mid += 1
            holder += mid
            list = list[mid:]
            mid = (len(list) - 1) // 2
        else:
            print("Value NONE")

    print(f"Index is = {mid + holder} \n")


Binary(7, L1)


# Example of using "break"
var = 10
while var > 0:
    print(f"Var value = {var}")
    var = var - 1
    if var == 5:
        print("Good bye!\n")
        break  # Terminates the while loop

# Another way to format it (Only for understanding purposes):
var = 10
while var > 0:
    if var == 5:
        print("Good bye!")
        break  # Terminates the while loop
    print(f"Var value = {var}")
    var = var - 1
