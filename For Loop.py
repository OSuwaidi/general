# بسم الله الرحمن الرحيم
# For loop examples:

L1 = ['omar', 'sultan', 'khalid']
L2 = ['lion', 'parrot', 'bear']
for lis, text in zip((L1, L2), ("names:", "\nanimals:")):  # --> "lis" calls [L1, L2], while "text" calls the strings "names:" the first time, then "animals:" the second time
    print(text)
    for i in lis:
        print(i)
print('\n\n')


for num in range(0, 5):
    print(str(5 - num) + ' little sheeps jumping on the bed, '
                         '1 fell off and bumped his head, '
                         'momma called the doctor and the doctor said, '
                         'no more sheeps jumping on the bed')
print('\n')

for num in range(5, 0, -1):
    print(str(num) + ' little sheeps jumping on the bed, '
                     '1 fell off and bumped his head, '
                     'momma called the doctor and the doctor said, '
                     'no more sheeps jumping on the bed')
print('\n')

########################################################################################################################
for x in range(1, 11):
    print("2*" + str(x), "=", 2 * x)

print('\n')

########################################################################################################################
a = "Dogs "
b = "Cats"
c = a + b
for index in range(0, len(c), 2):
    print(c[index], end="")  # The (end=""): executes the code in one line horizontally, instead of vertically
print("\n")

# Or you can simply use:
print(c[::2])  # Take everything in that string, but go in step sizes of 2
print(c[::-1], "\n")  # Reverse the order in that string

# Can also use while loop:
i = 0
while i < len(c):
    print(c[i], end="")
    i += 1
