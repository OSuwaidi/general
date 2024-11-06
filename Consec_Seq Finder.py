# بسم الله الرحمن الرحيم

# Objective: For each string in "strs", count the biggest number of ***consecutive*** substrings that exist in "dna" matching the strings in "strs".

dna = 'AAGGTAAGTTTAGAATATAAAAGGTGAGTTAAATAGAATAGGTTAAAATTAAAGGAGATCAGATCAGATCAGATCTATCTATCTATCTATCTATCAGAAAAGAGTAAATAGTTAAAGAGTAAGATATTGAATAGATCTAATGGAAAATATTGTTGGGGAAAGGAGGGATAGAAGG'
strs = ['AGATC', 'AATG', 'TATC']


def seq_finder(sequence, dna):
    start = 0  # Will allow us to skip scanned sequences
    counter = [0] * len(sequence)  # Create a list of zeros to store sequence occurrences
    for idx, seq in enumerate(sequence):  # Iterate over every entry in our sequence "strs"
        k = len(seq)
        holder = 0  # A temporary holder that will store #occurrences of *consecutive* sequences
        for i in range(start, len(dna)):  # For each sequence, iterate over our "dna" strand
            if dna[i:i+k] == strs[idx]:  # If match is found:
                holder += 1  # Increment our holder by 1
                while dna[i:i+k] == dna[i+k:i+k*2]:  # If our match has an identical match ahead (consecutively):
                    holder += 1  # Increment our holder by 1
                    i += k  # Start the next list indexing from our new match
                    start = i + 1  # To skip repetitive iterations over same matches
                if holder > counter[idx]:
                    counter[idx] = holder  # Only replace counter if new holder > old holder
                holder = 0  # Reset the holder when we existed our of our while loop (finished finding consecutives)
    return counter


print(seq_finder(strs, dna))
