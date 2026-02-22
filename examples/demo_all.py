"""
Demo: Run all ML algorithms from scratch on sample datasets.
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs, load_iris

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algorithms import (
    LinearRegression, LogisticRegression, KNN, DecisionTree,
    RandomForest, GaussianNaiveBayes, KMeans, PCA, SVM, NeuralNetwork
)


def separator(name):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")


# --- Regression ---
separator("Linear Regression")
X_reg, y_reg = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)
lr = LinearRegression(lr=0.01, n_iters=1000)
lr.fit(X_reg, y_reg)
print(f"  R2 Score: {lr.r2_score(X_reg, y_reg):.4f}")

# --- Classification datasets ---
X_clf, y_clf = make_classification(n_samples=300, n_features=4, n_redundant=0, random_state=42)

separator("Logistic Regression")
log_reg = LogisticRegression(lr=0.1, n_iters=1000)
log_reg.fit(X_clf, y_clf)
print(f"  Accuracy: {(log_reg.predict(X_clf) == y_clf).mean():.2%}")

separator("K-Nearest Neighbors (k=5)")
knn = KNN(k=5)
knn.fit(X_clf, y_clf)
print(f"  Accuracy: {(knn.predict(X_clf) == y_clf).mean():.2%}")

separator("Decision Tree")
dt = DecisionTree(max_depth=5)
dt.fit(X_clf, y_clf)
print(f"  Accuracy: {(dt.predict(X_clf) == y_clf).mean():.2%}")

separator("Random Forest (10 trees)")
rf = RandomForest(n_trees=10, max_depth=5)
rf.fit(X_clf, y_clf)
print(f"  Accuracy: {(rf.predict(X_clf) == y_clf).mean():.2%}")

separator("Gaussian Naive Bayes")
nb = GaussianNaiveBayes()
nb.fit(X_clf, y_clf)
print(f"  Accuracy: {(nb.predict(X_clf) == y_clf).mean():.2%}")

separator("Support Vector Machine")
svm = SVM(lr=0.001, lambda_param=0.01, n_iters=200)
svm.fit(X_clf, y_clf)
print(f"  Accuracy: {(svm.predict(X_clf) == y_clf).mean():.2%}")

# --- Clustering ---
separator("K-Means Clustering")
X_blobs, _ = make_blobs(n_samples=300, centers=4, random_state=42)
km = KMeans(k=4)
km.fit(X_blobs)
print(f"  Inertia: {km.inertia:.2f}")
print(f"  Cluster sizes: {[np.sum(km.labels == i) for i in range(4)]}")

# --- Dimensionality Reduction ---
separator("PCA")
X_iris, y_iris = load_iris(return_X_y=True)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_iris)
print(f"  {X_iris.shape} -> {X_reduced.shape}")
print(f"  Variance explained: {pca.explained_variance_ratio}")

# --- Neural Network ---
separator("Neural Network")
X_nn, y_nn = make_classification(n_samples=500, n_features=10, n_classes=3,
                                  n_informative=8, n_redundant=0, random_state=42)
nn = NeuralNetwork(layer_sizes=[32, 16], lr=0.1, n_iters=300)
nn.fit(X_nn, y_nn)
print(f"  Accuracy: {(nn.predict(X_nn) == y_nn).mean():.2%}")

print(f"\n{'='*50}")
print("  All algorithms ran successfully!")
print(f"{'='*50}")
