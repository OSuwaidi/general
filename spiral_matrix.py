# بسم الله الرحمن الرحيم و به نستعين

def spiral_matrix(matrix):
    spiral = []
    r = len(matrix)
    c = len(matrix[0])
    tot = r * c
    i = 0
    while len(spiral) != tot:
        if (rdiff := r - i * 2) == 2:  # even number of rows matrix
            spiral.extend(matrix[i][i:c - i] + matrix[r - i - 1][i:c - i][::-1])
        elif rdiff == 1:  # odd number of rows matrix
            spiral.extend(matrix[i][i:c - i])
        elif (c - i * 2) == 1:  # one column left in the matrix
            spiral.extend([row[i] for row in matrix[i:r - i]])
        else:
            last_first = [(row[c - i - 1], row[i]) for row in matrix[i + 1:r - i - 1]]
            last, first = zip(*last_first)
            spiral.extend(matrix[i][i:c - i] + list(last) + matrix[r - i - 1][i:c - i][::-1] + list(first)[::-1])
        i += 1
    return spiral


print(spiral_matrix([[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16]]))
