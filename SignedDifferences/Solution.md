# Solution

## Initial Observations
Let $D = (d_{ij})$ and let $`\hat{D} = (\hat{x}_i - \hat{x}_j)`$. Note that both matrices are skew-symmetric by design. Therefore, their difference is skew-symmetric.
Let $H := (d_{ij})_{i < j}$ and let $\hat{X} = (\hat{x}_i)$. 
The skew-symmetry implies that the sum of squares from the original problem satisfies:
```math
\sum_{ij} (d_{ij} - (\hat{x}_i - \hat{x}_j))^2 = 2\sum_{i < j} (d_{ij} - (\hat{x}_i - \hat{x}_j)^2) = 2\left\lVert A \hat{X} - H\right\rVert^2_2,
```
where $\left\lVert \cdot \right\rVert_2$ denotes the standard $\ell_2$-norm on $\mathbb{R}^n$ and $A$ is a design matrix that is to be determined.

## Reformulating The Problem
As per our initial observations, we can now think of the initial problem as a least-squares problem, subject to the constraint $\hat{x}_i \geq 0$ for all $i = 1, \cdots, n$, with design
matrix $A$ and target matrix $H$.

## Calculating The Design Matrix $$A$$
Note that the design matrix $A$ is of dimensions $n(n-1)/2 \times n$. We can easily infer the structure of this matrix. Indeed, this matrix transforms the vector
$$(\hat{x}_1, \cdots, \hat{x}_n)$$
to the vector of differences
```math
(\hat{x}_1 - \hat{x}_2, \cdots, \hat{x}_1 - \hat{x}_n, \hat{x}_2 - \hat{x}_3, \cdots, \hat{x}_2 - \hat{x}_n, \cdots, \hat{x}_{n-1} - \hat{x}_n)
```

Therefore, the first $n$ rows of $A$ consist of $n$-dimensional vectors whose first entry is $1$ and whose $j$-th entry is $-1$ for $j = 2, \cdots, n$; with the rest of the entries all being $0$. The next $n-1$ rows consist of $n$-dimensional vectors whose second entry is $1$ and whose $j$-th entry is $-1$ for $j = 3, \cdots, n$; with the rest of the entries all being $0$, and so on, until we get to the last row, whose last entry is $-1$, with the previous entry being $1$ and the rest being zeroes.

## Calculating The Target Matrix
The target matrix $H$ is simply the half-vectorization of the original matrix $D$ without the diagonal components (all of which are $0$). 
A careful observation of the pattern of $H$ shows how to construct it from $D$. Indeed, it suffices to see that its entries consist of the entries $d_{i, i+k}$ where $k = 1, \cdots, n-i-1$, with $i = 1, \cdots, n$.

## Solving The Problem
Once $A$ and $H$ are computed, we can simply compute $\hat{X}$ by solving the Non-Negative Least Squares problem. A number of implementations are available. We offer a solver based on a number of different libraries -- namely `CVXPY`, `scipy` and `scikit-learn`. The solutions offered by the last two seem to be identical, but `CVXPY` yields superior results in terms of residual errors. One can also regularize the problem in an effort to obtain better solutions.
