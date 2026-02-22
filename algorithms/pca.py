"""
Principal Component Analysis (PCA) from Scratch
=================================================
Linear dimensionality reduction using eigendecomposition of the covariance matrix.

Steps:
1. Center the data (subtract mean)
2. Compute covariance matrix
3. Compute eigenvectors and eigenvalues
4. Select top-k eigenvectors
5. Project data onto new subspace
"""

import numpy as np


class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalue descending
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio = self.explained_variance / eigenvalues.sum()
        return self

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    print(f"Original shape: {X.shape}")
    print(f"Reduced shape:  {X_reduced.shape}")
    print(f"Variance explained: {pca.explained_variance_ratio}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio):.2%}")
