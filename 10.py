import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the data (optional but recommended for PCA)
X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Initialize PCA with the desired number of components (in this case, 2)
pca = PCA(n_components=2)

# Fit and transform the data
X_pca = pca.fit_transform(X_standardized)

# Plot the results
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=70)
plt.colorbar(scatter, label='Target Class')
plt.title('PCA on Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
