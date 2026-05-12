# Linear Algebra for ML & AI — Cheatsheet

> Ch1 Vectors · Ch2 Matrices · Ch3 Linear Eqs · Ch4 Vector Spaces · Ch5 Eigenvalues · Ch6 Decompositions · Ch7 ML Apps

---

## Ch 1 — Vectors

**Vector & Operations**

`v ∈ ℝⁿ` · addition `u + v` · scaling `cv`

**Dot Product**

$$\mathbf{u} \cdot \mathbf{v} = \sum_i u_i v_i = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$$

**Norm** `‖v‖ = √(v·v)` &nbsp; **Unit** `v̂ = v / ‖v‖`

**Orthogonal** `u·v = 0` &nbsp; **Cosine similarity** `cos θ = (u·v) / (‖u‖ ‖v‖)`

**Projection**

$$\text{proj}_\mathbf{b}\mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{b} \cdot \mathbf{b}}\,\mathbf{b}$$

**Span & Linear Independence**

`span{v₁,…,vₖ}` = all linear combos  
Independent ⟺ `Σ cᵢvᵢ = 0 ⟹ cᵢ = 0 ∀i`

**NumPy**

```python
import numpy as np
u, v = np.array([1,2,3]), np.array([4,5,6])
dot  = np.dot(u, v)
norm = np.linalg.norm(v)
cos  = dot / (norm * np.linalg.norm(u))
proj = (np.dot(u,v) / np.dot(v,v)) * v
```

> **ML ›** Embeddings (word2vec, BERT) are vectors. Cosine sim = semantic similarity.

---

## Ch 2 — Matrices

**Definitions & Transpose**

`A ∈ ℝᵐˣⁿ` · `(Aᵀ)ᵢⱼ = Aⱼᵢ` · `(AB)ᵀ = BᵀAᵀ`

**Matrix Multiply**

$$(AB)_{ij} = \sum_k A_{ik}B_{kj} \qquad [m \times k] \cdot [k \times n] = [m \times n]$$

**Trace** `tr(A) = Σ Aᵢᵢ` &nbsp; **Det** `det[[a,b],[c,d]] = ad − bc`

`det(AB) = det(A) · det(B)`

**Inverse**

$$AA^{-1} = I \quad \text{exists iff } \det A \neq 0$$

**NumPy**

```python
C    = A @ B                # multiply
At   = A.T                  # transpose
det  = np.linalg.det(A)
Ainv = np.linalg.inv(A)
rank = np.linalg.matrix_rank(A)
tr   = np.trace(A)
```

> **ML ›** Every NN layer: `y = Wx + b` — matrix-vector product.

---

## Ch 3 — Linear Equations

**The System `Ax = b`**

- **Unique:** `rank(A) = n`, square system
- **No solution:** overdetermined, inconsistent
- **Infinite:** underdetermined

**LU Factorization**

$$A = LU \quad L\text{ lower-tri},\; U\text{ upper-tri}$$
Solve `Ly = b` then `Ux = y` — O(n²) each

**Least Squares — overdetermined (m > n)**

Minimise `‖Ax − b‖²`:

$$A^\top A\,x = A^\top b \quad \Rightarrow \quad x^* = (A^\top A)^{-1} A^\top b$$

**NumPy**

```python
x = np.linalg.solve(A, b)        # exact (square)
x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
from scipy.linalg import lu
P, L, U = lu(A)
```

> **ML ›** Linear regression solves least squares. `x*` = optimal weights.

---

## Ch 4 — Vector Spaces

**Four Subspaces of `A ∈ ℝᵐˣⁿ`, rank `r`**

| Subspace | Name | Dim |
|----------|------|-----|
| `C(A)` | column space ⊆ ℝᵐ | r |
| `N(A)` | null space ⊆ ℝⁿ | n − r |
| `C(Aᵀ)` | row space ⊆ ℝⁿ | r |
| `N(Aᵀ)` | left null ⊆ ℝᵐ | m − r |

`C(A) ⊥ N(Aᵀ)` &nbsp; `C(Aᵀ) ⊥ N(A)`

**Gram-Schmidt**

$$\mathbf{u}_i = \mathbf{a}_i - \sum_{j<i} \frac{\mathbf{a}_i \cdot \mathbf{q}_j}{\|\mathbf{q}_j\|^2}\,\mathbf{q}_j \qquad \mathbf{q}_i = \frac{\mathbf{u}_i}{\|\mathbf{u}_i\|}$$

**QR & Projection**

$$A = QR,\quad Q^\top Q = I$$
Projection matrix: `P = QQᵀ`, `P² = P`, `Pᵀ = P`

**NumPy**

```python
Q, R  = np.linalg.qr(A)
P     = Q @ Q.T            # projection matrix
b_hat = P @ b              # project onto C(A)
```

> **ML ›** Gram-Schmidt underlies QR algorithms & stable NN training.

---

## Ch 5 — Eigenvalues & Eigenvectors

**Definition**

$$A\mathbf{v} = \lambda\mathbf{v} \quad \mathbf{v} \neq \mathbf{0}:\text{ eigenvec},\; \lambda:\text{ eigenvalue}$$

Char. poly: `det(A − λI) = 0` · `tr(A) = Σλᵢ` · `det(A) = Πλᵢ`

**Diagonalisation**

$$A = P\Lambda P^{-1} \qquad \Lambda = \text{diag}(\lambda_1,\ldots,\lambda_n)$$
Powers: `Aᵏ = PΛᵏP⁻¹`

**Spectral Theorem (symmetric `A = Aᵀ`)**

$$A = Q\Lambda Q^\top,\quad Q^\top = Q^{-1}$$
(real λᵢ, orthogonal eigenvecs)

**Positive Definite:** all `λᵢ > 0` ⟺ `xᵀAx > 0 ∀x ≠ 0`

**NumPy**

```python
vals, vecs = np.linalg.eig(A)
vals, vecs = np.linalg.eigh(A_sym)  # symmetric
A3 = vecs @ np.diag(vals**3) @ vecs.T
```

> **ML ›** PCA uses covariance eigenvecs. PageRank = dominant eigenvec.

---

## Ch 6 — Matrix Decompositions

**SVD — Master Decomposition (any `A`)**

$$A = U\Sigma V^\top \quad U\text{: left sing. vecs},\; \Sigma\text{: singular vals},\; V\text{: right sing. vecs}$$

Best rank-k approx: `Aₖ = Σᵢ₌₁ᵏ σᵢ uᵢvᵢᵀ` (Eckart-Young)

`‖A‖₂ = σ₁` &nbsp; `‖A‖_F = √(Σσᵢ²)` &nbsp; `κ(A) = σ_max / σ_min`

**Comparison**

| Name | Form | Use |
|------|------|-----|
| LU | `PA = LU` | Solve `Ax = b` |
| QR | `A = QR` | Least squares |
| Cholesky | `A = LLᵀ` | Sym PD, covariance |
| EVD | `PΛP⁻¹` | Powers, spectral |
| SVD | `UΣVᵀ` | PCA, compression |

**NumPy**

```python
U, s, Vt = np.linalg.svd(A, full_matrices=False)
Ak = (U[:,:k] * s[:k]) @ Vt[:k,:]
from scipy.linalg import lu, cho_factor, cho_solve
P, L, U2 = lu(A)
c, low   = cho_factor(A_pd)
x        = cho_solve((c, low), b)
```

> **ML ›** SVD powers recommenders, image compression, LSA/NLP.

---

## Ch 7 — Applications in ML & AI

**PCA**

Centre `X̃ = X − x̄`, `C = (1/n) X̃ᵀX̃`

Eigen-decomp `C = QΛQᵀ` → project `Z = X̃Qₖ`

Via SVD: PCs = cols of `V` in `X̃ = UΣVᵀ`

**Linear Regression**

$$\hat{y} = Xw, \quad \mathcal{L} = \|Xw - y\|^2 \quad \Rightarrow \quad w^* = (X^\top X)^{-1} X^\top y$$

**Attention (Transformers)**

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**Matrix Factorization (Recommenders)**

`R ≈ UVᵀ`, `U ∈ ℝᵐˣᵏ`, `V ∈ ℝⁿˣᵏ`  
Min `Σ(i,j) (Rᵢⱼ − uᵢ·vⱼ)²` via ALS / SGD

**NumPy**

```python
from sklearn.decomposition import PCA
Z = PCA(n_components=2).fit_transform(X)
w = np.linalg.lstsq(X, y, rcond=None)[0]
U, s, Vt = np.linalg.svd(R, full_matrices=False)
R_hat = (U[:,:k] * s[:k]) @ Vt[:k,:]
```

> **ML ›** Everything in modern AI — regression, attention, PCA — is linear algebra.

---

## Quick Reference — Key Identities

**Transpose & Inverse**

$$(AB)^\top = B^\top A^\top \qquad (AB)^{-1} = B^{-1} A^{-1}$$

**Rank-Nullity**

$$\text{rank}(A) + \text{nullity}(A) = n$$

**Norms**

$$|\mathbf{u} \cdot \mathbf{v}| \leq \|\mathbf{u}\|\|\mathbf{v}\| \quad \text{(Cauchy-Schwarz)}$$
$$\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\| \quad \text{(Triangle)}$$

**Eigenvalues**

$$\text{tr}(A) = \sum \lambda_i \qquad \det(A) = \prod \lambda_i$$
$$A\mathbf{v} = \lambda\mathbf{v} \;\Rightarrow\; A^k\mathbf{v} = \lambda^k\mathbf{v}$$

**SVD / Conditioning**

$$\|A\|_2 = \sigma_{\max} \qquad \kappa(A) = \sigma_{\max} / \sigma_{\min}$$

**Projection**

$$P^2 = P,\quad P^\top = P,\quad \mathbf{e} = \mathbf{b} - P\mathbf{b} \perp C(A)$$

---

## NumPy linalg — Reference

```python
import numpy as np
np.zeros((m,n)); np.eye(n)       # construct
A @ B; A.T; np.dot(u,v)         # basic ops
np.linalg.norm(v)               # 2-norm
np.linalg.norm(A,'fro')         # Frobenius
np.linalg.solve(A, b)           # Ax=b (square)
np.linalg.lstsq(A, b)           # least squares
np.linalg.inv(A)                # inverse
np.linalg.pinv(A)               # pseudoinverse
np.linalg.det(A)                # determinant
np.linalg.matrix_rank(A)        # rank
np.trace(A)                     # trace
np.linalg.cond(A)               # condition number
np.linalg.eig(A)                # eigenvals/vecs
np.linalg.eigh(A)               # symmetric eig
np.linalg.svd(A)                # U, s, Vt
np.linalg.qr(A)                 # Q, R
np.linalg.cholesky(A)           # lower-tri L
```

---

*Linear Algebra for ML & AI · `01-vector.ipynb` → `07-applications-in-ml-and-ai.ipynb`*
