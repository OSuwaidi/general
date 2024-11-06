# بسم الله الرحمن الرحيم

import numpy as np  # Note: numpy is way faster than torch
import time
tp = time.perf_counter()

start = np.array([[2, 8, 3],
                  [1, 6, 4],
                  [7, ' ', 5]])

finish = np.array([[8, 1, 3],
                   [2, ' ', 4],
                   [7, 6, 5]])

collection = []  # List of arrays to check against our target/desired state ("finish")
visited = [start]  # Append the state we initially started with ("start") so that we don't append it again
n = 0


def match(parent, target):
    global n, collection
    n += 1
    current = parent.copy()  # Create a copy of our parent state, so that changes on "current" won't affect parent state
    temp = parent.copy()  # To find the location of ' '
    r, c = np.where(parent == ' ')  # Gives coordinates/axes of ' ' location

    if c < 2:
        current[r, c] = current[r, c + 1]
        current[r, c + 1] = temp[r, c]
        if not (current == np.stack(visited)).all(axis=(1, 2)).any():  # "axis=1" is comparing columns, "axis=2" is comparing rows, ***"axis=(1,2)" is comparing both rows and columns (compares entire matrices)***
            collection.append(current)  # If "current" state was never visited before, add it to "collection" to check whether it's our target or not

    current = parent.copy()
    if c > 0:
        current[r, c] = current[r, c - 1]
        current[r, c - 1] = temp[r, c]
        if not (current == np.stack(visited)).all(axis=(1, 2)).any():
            collection.append(current)

    current = parent.copy()
    if r > 0:
        current[r, c] = current[r - 1, c]
        current[r - 1, c] = temp[r, c]
        if not (current == np.stack(visited)).all(axis=(1, 2)).any():
            collection.append(current)

    current = parent.copy()
    if r < 2:
        current[r, c] = current[r + 1, c]
        current[r + 1, c] = temp[r, c]
        if not (current == np.stack(visited)).all(axis=(1, 2)).any():
            collection.append(current)

    for state in collection:
        visited.append(state)
        if (state == target).all():  # Or: "if False not in (state == target):"
            return state
    collection = []  # Reset "collection" since no state in collection was our target

    return match(visited[n], target)  # Starts with n=1, since n=0 is the initial state we started with ("start")


print(match(start, finish))
print(f'Time = {time.perf_counter() - tp:.4} \nSteps = {n}')
