# بسم الله الرحمن الرحيم
# Combines elements from sequences based on ascending order element-wise (comparatively)

def interleaved(seq1, seq2):
    i = 0
    j = 0
    res = []

    while i < len(seq1) and j < len(seq2):
        if seq1[i] < seq2[j]:
            res.append(seq1[i])
            i += 1
            if i == len(seq1):  # If we explored all of seq1 (reached the end)
                for num in seq2[j:]:  # Explore the rest of seq2
                    res.append(num)  # Append the rest
                return res
        if seq2[j] <= seq1[i]:
            res.append(seq2[j])
            j += 1  # Assuming seq2 is > seq1

    return res


print(interleaved([-7, -2, 1], [-4, -3, -5, 8]))
