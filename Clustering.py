# بسم الله الرحمن الرحيم

from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

plt.style.use('seaborn')
n = 500
blobs = datasets.make_blobs(n_samples=n, random_state=6)  # "random_state=#" will have a specific plot (distribution) called everytime
points, target_clusters = blobs

x = points[:, 0]  # All rows from the first column (x values)
y = points[:, 1]  # All rows from the second column (y values)
plt.scatter(x, y)
plt.show()

clusters = KMeans(n_clusters=3, max_iter=10).fit_predict(points)
colors = ['b', 'g', 'r', 'μ', 'm', 'y']

for i in range(len(clusters)):
    col = colors[clusters[i]]
    plt.scatter(*points[i], c=col)  # Recall: "*points" expands the list into its components (x, y in this case)
plt.show()
