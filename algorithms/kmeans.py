"""
K-Means Clustering from Scratch
================================
Partitions data into k clusters by minimizing within-cluster sum of squares.

Algorithm:
1. Initialize centroids using k-means++
2. Assign each point to nearest centroid
3. Update centroids as mean of assigned points
4. Repeat until convergence
"""

import numpy as np


class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.inertia = None

    def _kmeans_plus_plus(self, X):
        n_samples = X.shape[0]
        centroids = [X[np.random.randint(n_samples)]]

        for _ in range(1, self.k):
            distances = np.array([
                min(np.sum((x - c) ** 2) for c in centroids)
                for x in X
            ])
            probs = distances / distances.sum()
            idx = np.random.choice(n_samples, p=probs)
            centroids.append(X[idx])

        return np.array(centroids)

    def _assign_clusters(self, X):
        distances = np.array([
            np.sum((X - c) ** 2, axis=1) for c in self.centroids
        ]).T
        return np.argmin(distances, axis=1)

    def fit(self, X):
        self.centroids = self._kmeans_plus_plus(X)

        for _ in range(self.max_iters):
            self.labels = self._assign_clusters(X)

            new_centroids = np.array([
                X[self.labels == i].mean(axis=0) if np.sum(self.labels == i) > 0
                else self.centroids[i]
                for i in range(self.k)
            ])

            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids

        self.inertia = sum(
            np.sum((X[self.labels == i] - self.centroids[i]) ** 2)
            for i in range(self.k)
        )
        return self

    def predict(self, X):
        return self._assign_clusters(X)


if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)
    model = KMeans(k=4)
    model.fit(X)
    print(f"K-Means Inertia: {model.inertia:.2f}")
    print(f"Cluster sizes: {[np.sum(model.labels == i) for i in range(4)]}")
