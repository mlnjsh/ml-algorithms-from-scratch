# ML Algorithms from Scratch

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-only-013243?logo=numpy)
![License](https://img.shields.io/badge/License-MIT-green)

> Machine Learning algorithms implemented from scratch using only NumPy. No scikit-learn, no black boxes — just pure Python and math.

Every algorithm includes the mathematical derivation, a clean implementation with a scikit-learn-compatible API (`fit`/`predict`), and a runnable demo.

---

## Algorithms

| Algorithm | Type | File | Key Concepts |
|-----------|------|------|-------------|
| Linear Regression | Regression | [`linear_regression.py`](algorithms/linear_regression.py) | Gradient descent, closed-form (normal equation), MSE, R-squared |
| Logistic Regression | Classification | [`logistic_regression.py`](algorithms/logistic_regression.py) | Sigmoid, cross-entropy loss, decision boundary |
| K-Nearest Neighbors | Classification | [`knn.py`](algorithms/knn.py) | Distance metrics, majority voting, curse of dimensionality |
| Decision Tree | Classification | [`decision_tree.py`](algorithms/decision_tree.py) | Information gain, Gini impurity, recursive splitting |
| Random Forest | Classification | [`random_forest.py`](algorithms/random_forest.py) | Bootstrap aggregation, feature subsampling, ensemble |
| Naive Bayes | Classification | [`naive_bayes.py`](algorithms/naive_bayes.py) | Bayes theorem, Gaussian likelihood, class priors |
| K-Means | Clustering | [`kmeans.py`](algorithms/kmeans.py) | K-means++ init, Lloyd's algorithm, inertia |
| PCA | Dimensionality Reduction | [`pca.py`](algorithms/pca.py) | Eigendecomposition, variance explained, projection |
| SVM | Classification | [`svm.py`](algorithms/svm.py) | Hinge loss, subgradient descent, margin maximization |
| Neural Network | Classification | [`neural_network.py`](algorithms/neural_network.py) | Backpropagation, activation functions, weight initialization |

---

## Quick Start

```bash
git clone https://github.com/mlnjsh/ml-algorithms-from-scratch.git
cd ml-algorithms-from-scratch
pip install numpy matplotlib

# Run all demos
python examples/demo_all.py

# Or run individual algorithms
python -m algorithms.linear_regression
python -m algorithms.neural_network
```

## Example

```python
from algorithms.logistic_regression import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=2, random_state=42)
model = LogisticRegression(lr=0.01, n_iters=1000)
model.fit(X, y)
predictions = model.predict(X)
accuracy = (predictions == y).mean()
print(f"Accuracy: {accuracy:.2%}")
```

## Requirements

- Python 3.8+
- NumPy (core dependency)
- matplotlib (optional, for visualizations)
- scikit-learn (optional, only for comparison/datasets in demos)

## Project Structure

```
ml-algorithms-from-scratch/
  algorithms/
    __init__.py
    linear_regression.py
    logistic_regression.py
    knn.py
    decision_tree.py
    random_forest.py
    naive_bayes.py
    kmeans.py
    pca.py
    svm.py
    neural_network.py
  examples/
    demo_all.py
  requirements.txt
  README.md
```

## License

MIT


---

## Contributors & Domain Experts

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/mlnjsh">
        <img src="https://github.com/mlnjsh.png" width="80px;" alt="Milan Amrut Joshi"/><br />
        <sub><b>Milan Amrut Joshi</b></sub>
      </a><br />
      <sub>Project Author</sub>
    </td>
    <td align="center">
      <a href="https://github.com/eriklindernoren">
        <img src="https://github.com/eriklindernoren.png" width="80px;" alt="Erik Linder-Noren"/><br />
        <sub><b>Erik Linder-Noren</b></sub>
      </a><br />
      <sub>ML From Scratch implementations</sub>
    </td>
    <td align="center">
      <a href="https://github.com/rushter">
        <img src="https://github.com/rushter.png" width="80px;" alt="Egor Pakhomov"/><br />
        <sub><b>Egor Pakhomov</b></sub>
      </a><br />
      <sub>ML from scratch in Python</sub>
    </td>
  </tr>
</table>
