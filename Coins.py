# بسم الله الرحمن الرحيم
# You have some amount of money that you want to exchange in for coins that come in the following values: [1, 2, 5, 10, 20, 50, 100, 500]
# What is the least number of coins of such values that you can get, that satisfy the amount of money you turned over?

import time

start = time.time()


def exchange(amount):
    coins = [1, 2, 5, 10, 20, 50, 100, 500]
    n = len(coins) - 1
    my_coins = []
    while n >= 0:
        while amount >= coins[n] and amount != 0:  # If amount = 0, this implies we have covered the amount with coins, therefore to need to continue searching and iterating
            amount -= coins[n]
            my_coins.append(coins[n])
        n -= 1
    print("You get =", my_coins)


exchange(950)
print("Time = {:.3} seconds".format(time.time() - start))
