import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate a sample dataset
X, y= make_blobs(n_samples=300, centers=4, random_state=42)

# Visualize the dataset without specifying colormapping
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title('Generated Dataset')
plt.show()

# Apply K-means clustering
model = KMeans(n_clusters=4, random_state=42, n_init=10)
model.fit(X)

# Get cluster centers and labels
cluster_centers = model.cluster_centers_
labels = model.labels_

# Visualize the clustered dataset
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
print()
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title('K-means Clustering')
plt.legend()
plt.show()
