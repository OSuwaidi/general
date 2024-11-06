# بسم الله الرحمن الرحيم

import numpy as np
import time
tp = time.perf_counter()

start = np.array([[2, 8, 3],
                  [1, 6, 4],
                  [7, ' ', 5]])

finish = np.array([[8, 1, 3],
                   [2, ' ', 4],
                   [7, 6, 5]])
children = {'left': [], 'right': [], 'down': [], 'up': []}
parents = {'left': [], 'right': [], 'down': [], 'up': []}
step = 0
n = 0


def match(parent, target, flag=None):
    global n
    n += 1
    global children, parents, step  # Global these variables; so that when we adjust/alter them inside the function, they also get adjusted/altered outside the function
    current = parent.copy()  # To change our "parent" state/board
    temp = parent.copy()  # To find the location of ' '
    r, c = np.where(parent == ' ')  # Gives coordinates/axes of ' ' location
    directions = ['right', 'left', 'up', 'down']
    if flag:  # *** Note: "if (anything)" --> True, UNLESS "anything" was "False", "None" or "''" or "0" ***
        directions.remove(flag)

    if 'right' in directions and c < 2:
        current[r, c] = current[r, c + 1]
        current[r, c + 1] = temp[r, c]
        children['left'].append(current)  # Add key ('left') into "children" dictionary with value = "current"

    current = parent.copy()
    if 'left' in directions and c > 0:
        current[r, c] = current[r, c - 1]
        current[r, c - 1] = temp[r, c]
        children['right'].append(current)

    current = parent.copy()
    if 'up' in directions and r > 0:
        current[r, c] = current[r - 1, c]
        current[r - 1, c] = temp[r, c]
        children['down'].append(current)

    current = parent.copy()
    if 'down' in directions and r < 2:
        current[r, c] = current[r + 1, c]
        current[r + 1, c] = temp[r, c]
        children['up'].append(current)

    for lis in children.values():
        for child in lis:
            if False not in (child == target):  # Or: (child == target).all():  --> Will only be applicable if all elements in "children" were arrays
                return child  # Without "returning" your recursive function, this value would be "None", because the variable "child" would only be stored in that last function called (last leaf), but not in the main function (seed)
    for key in children:  # Add all newly created children into the "parents" category/dictionary since none of them were our target
        for value in children[key]:  # To access elements within the list inside [key]
            parents[key].append(value)
    children = {'left': [], 'right': [], 'down': [], 'up': []}  # Clear out exiting children since none were our target, and so that when we add "children" into "parents" (above) we don't add repeating children
    dirs = ['left', 'right', 'down', 'up']
    if step > 3:
        step = 0
    position = dirs[step]
    while not parents[position]:  # *** Note: "if *not* (anything)" --> False, UNLESS "anything" was "False", "None" or "''" ***
        step += 1
        if step > 3:
            step = 0
        position = dirs[step]
    new_parent = parents[position].pop(0)  # Remove the first entry from the list within specified ("position") key
    step += 1
    return match(new_parent, target, flag=position)  # *** Have to "return" the recursive function


print(match(start, finish))
print(f'Time = {time.perf_counter() - tp:.4} \nSteps = {n}')
