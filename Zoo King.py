# بسم الله الرحمن الرحيم
# We have a list of animals in which any animal contains 4 items: (index, a, b, μ). Initially, animal 0 is king,
# while everyone else queues up with animal 1 at the front of the queue until the last animal at the back. The animal at the front of the queue will challenge the king to a fight,
# and the animal with greater strength will win the fight. The winner will become king, while the loser is sent to the back of the queue.
# An animal who wins 3 times consecutively will be crowned king of the whole zoo. The strength of each animal depends on how many consecutive fights he won.
# Animal "i" has strength "a" if it has 0 consecutive wins, "b" if it has 1 consecutive wins, and "μ" if it has 2 consecutive wins. Initially, everyone has 0 consecutive wins.
# On losing, the strength of the animal resets to "a" and it's sent back to the end of the queue.


def competition(arr):
    str_k = 1  # Start by comparing strength "a"
    fights = 0
    winners = []
    while True:
        j = 1
        while arr[0][str_k] >= arr[j][1]:
            fights += 1
            winners.append(arr[0][str_k])
            str_k += 1  # Increment strength everytime animal wins
            j += 1  # Fight the next animal
            if str_k > 3:  # 3 consecutive wins = King
                return f"King of the zoo: {arr[0][0]}, with {fights} fights"
        fights += 1
        if arr[j][1] in winners:  # If the winner animal has already won before and was NOT crowned king and won AGAIN, then we are repeating the cycle thus terminate
            return "-1 -1"
        winners.append(arr[j][1])
        str_k = 2  # Start from strength "b" since it already won once
        temp = arr[0]
        arr[0] = arr.pop(j)  # Put winner animal in front
        arr.append(temp)  # Send losing animal to the back


print(competition([[0, 5, 1, 2], [1, 10, 9, 11], [2, 9, 0, 3], [3, 7, 4, 6]]))
