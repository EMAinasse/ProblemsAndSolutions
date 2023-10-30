# Problem Statement

Suppose we are given noisy observations $d_{ij}$ of the signed differences between $n$ numbers $x_1, \cdots, x_n$ -- i.e. $d_{ij} \approx x_i - x_j$.
We wish to estimate the estimators $`\hat{x}_1, \cdots, \hat{x}_n`$ that minimize:
$$\sum_{i, j} (d_{ij} - (\hat{x}_i - \hat{x}_j))^2$$
subject to the constraint $\hat{x}_i = 0$.

The data is given as a skew-symmetric matrix $D = (d_{ij})$.
