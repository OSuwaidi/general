# بسم الله الرحمن الرحيم

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from matplotlib import pyplot as plt

# Using k means on text:
txt = open('test.txt')
vectorizer = TfidfVectorizer()  # Convert a collection of raw (text) documents into a matrix
matrix = vectorizer.fit_transform(txt)
print(matrix)

kmeans = KMeans(n_clusters=12, n_init=10, init="k-means++", max_iter=100)
clusters = kmeans.fit_predict(matrix)
print(f'Classification of sample points: {clusters}\n')  # Prints to which centroid each sample went to

terms = list(vectorizer.vocabulary_)
print(f'The different unique terms are: {terms}\n')

centroids = kmeans.cluster_centers_
centroid_list = np.zeros(len(centroids), dtype=int)
for i in clusters:
    centroid_list[i] += 1
print(f'How many samples went to each centroid: {centroid_list}\n\n')


# Using k means on sample points:
data = np.array([[3, 1.5, 1], [3.5, 0.5, 1], [4, 1.5, 1], [5.5, 1, 1], [1, 1, 0], [2, 1, 0], [2, 0.5, 0], [3, 1, 0]])
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(data)
print(kmeans.cluster_centers_)
centers = sorted(kmeans.cluster_centers_, key=lambda x: x[2])  # Sort in ascending order based on the last index (first cluster (0) comes first)
for sample in data:
    plt.scatter(*sample[:2], c='red' if sample[2] == 1 else 'blue')
plt.scatter(*centers[0][:2], c='blue', marker='*', s=500)
plt.scatter(*centers[1][:2], c='red', marker='*', s=500)
plt.grid()
plt.axis([0, 6, 0, 2])
plt.show()
