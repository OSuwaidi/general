# بسم الله الرحمن الرحيم

from matplotlib import pyplot as plt
from math import sqrt, log  # Faster than numpy
import numpy as np  # Note: The statistics library is VERY slow, it's meant for precision and accuracy, not efficiency
import matplotlib

matplotlib.use('TKAgg')
plt.style.use('seaborn')  # Will apply to all plots generated from other scripts if you imported the functions from this script file


def dbscam(data, radius=1):
    radius **= 2
    data = list(data)
    list_of_clusters = []  # A list containing a list of our clustered points
    while data:  # While we still have samples in our data
        cluster = [data.pop()]  # Start a new cluster group with some random point as center initially
        for center in cluster:  # Scan and assign all the points around each center in our cluster group
            for i, point in enumerate(data):
                if np.sum((point - center)**2) < radius:
                    cluster.append(data.pop(i))
        list_of_clusters.append(cluster)
    return list_of_clusters


def mean_calc(lis):
    return sum(lis)/len(lis)


def dist(p1, p2):
    return sqrt(sum([(a-b)**2 for a, b in zip(p1, p2)]))


def dbscam_robust(data, radius=2, noise_f=1, refine=False):  # "noise_f": noise factor (the higher, the more noise is tolerated)
    if type(data[0]) is not list:
        data = [list(p) for p in data]
    clone = data[:]  # Now manipulating "clone" itself won't affect original list "data", BUT manipulating ***elements within the entries*** of "clone" WILL affect "data" and vice-versa since it's a list of lists, and the inner lists still share the same memory!!!
    g = 0  # To change the cluster number for each unique group/cluster
    significance = len(data)*(5/100)  # Require at least 5% of my data before performing statistical inference
    while clone:  # While the list is NOT empty
        g += 1
        history = []
        visited = [clone.pop()]  # Can alternatively pop the last/first index
        for v in visited:
            for point in clone:
                metric = dist(v, point)
                if metric <= radius:
                    if len(history) <= significance:  # Decreasing the minimum history length will make the clustering even more robust to noise, but very "picky"
                        history.append(metric)
                        visited.append(point)
                        clone.remove(point)  # So that no point gets assigned to more than one cluster/group (first come first serve). Note: "del" is much faster than ".remove()"!!!
                    else:
                        group_std = np.std(history) + 1e-7
                        point_std = (metric - mean_calc(history)*noise_f) / group_std  # Maybe do this after every "n" steps to reduce computation
                        if point_std <= group_std:
                            history.append(metric)
                            visited.append(point)
                            clone.remove(point)  # So that no point gets assigned to more than one cluster/group (first come first serve). Note: "del" is much faster than ".remove()"!!!
        if len(visited) > significance * (-log(noise_f + 0.1) + 1):  # Increase the required number of samples as robustness increases ==> "noise_f" decreases
            if refine:
                mean = np.mean(visited, 0)
                avg_dist = np.mean(history)
                std = np.std(history) + 1e-7
            for v in visited:
                if refine and dist(v, mean) >= avg_dist / std:  # Conditional statements are read from left to right, hence even if "mean" is not defined, it would terminate since "refine" would be "False" and "mean" is only defined when "refined" is True
                    v.append(0)
                    continue  # Go to the next entry in "visited" so that you won't append into "v" more than once
                v.append(g)
        else:
            for v in visited:
                v.append(0)  # "0" represents the noisy datapoints (we could also delete if we didn't want to see/use the noisy data)
    return data


def main():
    dataset = datasets.make_blobs(n_samples=500, random_state=7)
    points, clusters = dataset
    colors = [np.zeros(3)] + [np.random.rand(3) for _ in range(5)]  # The "(0, 0, 0)" in the first index is for the "noisy" datapoints
    grouped = dbscam(points)
    plt.title('DBscam', fontsize=18)
    for x, y, c in grouped:
        plt.scatter(x, y, color=colors[c])
    plt.show()

    grouped = dbscam_robust(points)
    plt.title('Robust DBscam', fontsize=18, c='r')
    for x, y, c in grouped:
        plt.scatter(x, y, color=colors[c])
    plt.show()


if __name__ == '__main__':  # Such that when you import a function from this script, the whole script won't run automatically
    from sklearn import datasets
    main()
