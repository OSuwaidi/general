# بسم الله الرحمن الرحيم
# Find the factors of a number, figure out if the factor is prime or not, make a tuple of those prime factors

inp = int(input('Enter the number: '))
prime = []
factors = [1]  # We add 1 here because 1 is always a factor of any number


def find(inp):
    factors.append(inp)  # Add the number it self, because every number is a factor of itself
    i = 2
    flag = 1
    while i < inp:
        if inp % i == 0 and i == 2:  # Find all the factors of "inp", not necessarily prime
            prime.append(i)
            flag = 0
        elif inp % i == 0:
            flag = 0  # Create a flag to check if the number had a any factors other than 1 and itself, if yes ==> Not prime
            prime_check = []  # Check list to determine if our factor "i" is prime or not
            for n in range(2, i):  # We started from 2, because every prime will be divisible by 0 and 1, and ended at "i-1", because every prime number is also a factor of itself
                prime_check.append(i % n)  # Check if our factor "i" is divisible by every entry of list "L" or not if 0 not in test:  # If there are no zeros in our "test" list, that means that our factor "i" was NOT divisible by any entry in list "L". There for it is prime!
            if 0 not in prime_check:  # If there are no zeros in our "test" list, that means that our factor "i" was NOT divisible by any number from 2 to (i-1). There for it is prime!
                prime.append(i)  # Append that "i" because it is a prime
        i += 1

    if flag == 1 and inp != 1 and inp != 0:  # We used the flag outside of the while loop, because if we were to print the below statement inside the while loop, it would print a lot of times
        print(f"{inp} is prime")  # Since i > 1, #1 will not enter the while loop, and will come out as f = 1 (making it a prime), therefore we added the "and inp!=1" statement to counter this, and have it as f = 0
    else:
        print(f"{inp} is not prime")


find(inp)
print(f"Factors = {sorted(factors)}")
print(f"Primes = {prime}")
