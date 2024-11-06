# بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ و به نستعين

import numpy as np


# The probability using Bayes Rule that the *achieved* statistical significance is *actually* significant!!!
# "args" represents the *(1 - p_value)* of each run that resulted in a statistical significance
# P(AS|GS) = (P(GS|AS) * P(AS)) / P(GS)
# P(AS|GS) = (1 - p) * α / (p * (1 - α) + (1 - p) * α)
def probability_of_significance(p_values: list[float] | tuple[float], significance_level: float = 0.05):
    num = np.prod(np.subtract(1, p_values)) * significance_level
    den = np.prod(p_values) * (1 - significance_level) + np.prod(np.subtract(1, p_values)) * significance_level
    return num/den


p = probability_of_significance([0.07])
print(p)
