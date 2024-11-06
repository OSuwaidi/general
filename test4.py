# بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ و به نستعين

import pandas as pd
import numpy as np

data = np.random.randint(0, 10, (100, 3))
df = pd.DataFrame(data, columns=['a', 'b', 'c'])

from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
print(df[['a', 'b']])
print(df["c"])
scores = cross_validate(clf, df[['a', 'b']], df['c'], scoring='accuracy', cv=2)
