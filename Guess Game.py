# بسم الله الرحمن الرحيم
# Guessing Game Challenge:

# Write a program that picks a random integer from 1 to 100, and has players guess the number. The rules are:

# If a player's guess is less than 1 or greater than 100, say "OUT OF BOUNDS"
# On a player's first turn, if their guess is within 10 of the number, return "WARM!"; else if their guess is further than 10 away from the number, return "COLD!"
# On all subsequent turns, if a guess is closer to the number than the previous guess return "WARMER!"; if not, return "COLDER!"
# When the player's guess equals the number, tell them they've guessed correctly and HOW MANY guesses it took them!

import torch


def guess():
    target = torch.randint(1, 101, (1, 1))
    i = 0  # pseudo guess
    tries = 0
    memory = 0
    while i != target:
        i = input('What is your guess? ')
        if not i.isdigit():
            print('Please enter a valid number')
        elif int(i) < 1 or int(i) > 100:
            print("Guess is out of bounds")
        elif i == memory:
            print("You didn't change your guess")
        else:
            tries += 1
            i = int(i)
            if i == target:
                break
            else:
                if tries == 1:
                    if abs(target - i) <= 10:
                        print('Warm!')
                    else:
                        print('Cold')
                else:
                    if abs(target - i) > abs(target - memory):
                        print('Colder')
                    else:
                        print('Warmer!')
                memory = i
    print(f"\nCorrect! You got it in {tries} tries!")


guess()
