# Linear Algebra for Machine Learning & AI

A hands-on, visual tutorial series covering the linear algebra concepts every ML/AI practitioner needs. Each chapter is a self-contained Jupyter notebook with NumPy code, matplotlib/Plotly visualizations, and ML connections throughout.

## Chapters

| # | Notebook | Topics |
|---|----------|--------|
| 1 | [Vectors](01-vector.ipynb) | Vector ops, dot product, unit vectors, orthogonality, span, linear independence |
| 2 | [Matrices](02-matrices.ipynb) | Matrix ops, transpose, inverse, determinant, rank, linear transformations |
| 3 | [Systems of Linear Equations](03-systems-of-linear-equations.ipynb) | Gaussian elimination, LU, least squares, overdetermined / underdetermined systems |
| 4 | [Vector Spaces & Orthogonality](04-vector-spaces-and-orthogonality.ipynb) | Basis, dimension, four fundamental subspaces, projections, Gram-Schmidt, QR |
| 5 | [Eigenvalues & Eigenvectors](05-eigenvalues-and-eigenvectors.ipynb) | Characteristic polynomial, diagonalization, spectral theorem, matrix powers |
| 6 | [Matrix Decompositions](06-matrix-decompositions.ipynb) | LU, QR, Cholesky, SVD — theory, NumPy, and when to use each |
| 7 | [Applications in ML & AI](07-applications-in-ml-and-ai.ipynb) | PCA, linear regression, recommender systems, word embeddings, neural networks |

## Setup

Requires Python 3.12+. Install dependencies with [uv](https://github.com/astral-sh/uv) (recommended) or pip:

```bash
# uv
uv sync

# pip
pip install numpy matplotlib scipy scikit-learn plotly seaborn pandas pillow ipykernel
```

Then open the notebooks:

```bash
jupyter lab
```

## Target Audience

Students, software engineers, and data scientists who want to build strong linear algebra intuition for ML/AI work. Familiarity with basic Python and NumPy is assumed; no prior linear algebra required.
