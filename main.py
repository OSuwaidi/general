# بسم الله الرحمن الرحيم و به نستعين

from sklearn.datasets import make_blobs
from DBscam import dbscam_robust
import numpy as np
from matplotlib import pyplot as plt


def dbscam_robusttt(data, radius=2, robustness=1):  # As robustness, less noise is tolerated
    radius **= 2
    data = list(data)
    flag = True
    list_of_clusters = []  # A list containing a list of our clustered points
    noise = []
    significance = len(data)*0.05  # Require at least 5% of my data before performing statistical inference
    while data:  # While we still have samples in our data
        history = []
        cluster = [data.pop()]  # Start a new cluster group with some random point as center initially
        for center in cluster:  # Scan and assign all the points around each center in our cluster group
            for i, point in enumerate(data):
                dist = np.sum((point - center)**2)
                if dist < radius:
                    if len(history) < significance:  # Decreasing the minimum history length will make the clustering even more robust to noise, but very "picky"
                        history.append(dist)
                        cluster.append(data.pop(i))
                    else:
                        group_std = np.std(history) + 1e-7
                        point_std = (dist - np.mean(history) * robustness) / group_std  # Maybe do this after every "n" steps to reduce computation
                        if point_std <= group_std:
                            history.append(dist)
                            cluster.append(data.pop(i))

        if len(cluster) > significance * (-np.log(robustness + 0.1) + 1):
            list_of_clusters.append(cluster)
        else:
            noise.append(cluster)

    list_of_clusters.append(noise)
    return list_of_clusters


dataset = make_blobs(n_samples=500, random_state=7)
data, clusters = dataset
clustered = dbscam_robusttt(data, robustness=0.5)
colors = [np.random.rand(3) for _ in range(len(clustered) - 1)] + [np.zeros(3)]
for i in range(len(clustered)):
    try:
        x, y = zip(*clustered[i])
    except:
        plt.scatter(x, y, color=colors[i])
plt.show()

grouped = dbscam_robust(data, noise_f=0.5)
plt.title('Robust DBscam', fontsize=18, c='r')
colors = [np.zeros(3)] + [np.random.rand(3) for _ in range(len(clustered) - 1)]
for x, y, c in grouped:
    plt.scatter(x, y, color=colors[c])
plt.show()
