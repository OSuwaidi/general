# بسم الله الرحمن الرحيم

# Find the probability of getting 5 or more heads from 7 tosses of a biased coin that settles on heads 60% of the time
# --> (x = #of_heads), *** P(x>=5 | 7 tosses) = ? ***
# Recall: P(x>=5 | 7 tosses) = The probability of a certain event/outcome "x" happening, times the number of *unique* occurrences/configurations of that event/outcome
# Or: P(x>=5 | 7 tosses) = The probability of a certain event/outcome "x" happening, times the number of different configurations that result in the same event/outcome
# P(x>=5 | 7 tosses) = ∑(x=5-->x=7) P(x) * #(x)

import random


def toss():
    return random.choices('HT', weights=(0.6, 0.4), k=7).count('H') >= 5  # Will return "True" or "False" ("True" if heads count was 5 or greater in 7 tosses)
                                # Or we can use: "cum_weights=(0.60, 1.00)" (same probability distribution as above)


# Find the average number of times it was "True" (our biased coin giving 5 heads or more in 7 tosses) --> (# of times event was "True" / total # of events)
# Note: number of states where x >= 5 == 7!/(5!⋅2!) + 7!/(6!⋅1!) + 7!/(7!⋅0!)
# Note: total number of possible states == 2**7
N = 10000
print(f"P(x>=5 |7 tosses) = {sum([toss() for _ in range(N)]) / N :.3}")


# For the same biased coin with the same number of tosses, what is the *expected* value of getting heads?
# --> (x = #of heads), *** E(x | 7 tosses) = ? ***
# Recall: E(x | 7 tosses) = The probability of outcome/event "x", times the value of the outcome "x", for all possible values of "x"!
# E(x | 7 tosses) = ∑(x=min-->x=max) P(x) * x

def toss():
    return random.choices('HT', weights=(0.6, 0.4), k=7).count('H')


print(f"E(x=#heads | 7 tosses) = {sum([toss() for _ in range(N)]) / N :.3}")
