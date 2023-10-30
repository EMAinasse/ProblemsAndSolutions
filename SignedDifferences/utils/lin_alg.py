"""Auxiliary linear algebra functions."""

import numpy as np

def half_vectorize(matrix : np.ndarray, triangular : str = 'upper', diag : bool = False) -> np.array:
    """
    A function to compute the upper or lower half-vectorization of an array; including or excluding the diagonal.

    Parameters
    ----------
    - matrix: np.ndarray
        The symmetric matrix to be vectorized.
    - triangular: str
        A string to indicate which triangular half of the matrix to consider: upper (by default) or lower.
    - diag: bool
        A boolean value to indicate whether or not the diagonal components are to be included in the vectorization (False by default).
    
    Returns
    -------
    - matrix_vec: np.array
        The vector representing the vectorization of the original matrix, as per the options specified.
    """

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Your 2D-array must be a square matrix.")
        
    if triangular not in ['upper', 'lower']:
        raise ValueError("The `triangular` parameter can either be `upper` or `lower`. Please choose one of these two values.")
    
    n = len(matrix)
    
    m = n * (n-1) // 2 if not diag else n * (n+1) // 2
    
    matrix_vec = np.zeros(m)
    
    diag_offset = 1 - int(diag)
    
    if triangular == 'upper':
        sum_index = -len(matrix) - (1 - diag_offset)

        for i in range(len(matrix) - diag_offset):

            sum_index += len(matrix) + (1 - diag_offset) - i

            for j in range(len(matrix) -i -diag_offset):

                matrix_vec[sum_index + j] = matrix[i][i + j + diag_offset]
    else:
        sum_index = -1

        for i in range(len(matrix) - diag_offset):

            sum_index += i+1

            for j in range(i+1):

                matrix_vec[sum_index - j] = matrix[i + diag_offset][i - j]        
    
    return matrix_vec

def diff_matrix(X : np.array) -> np.ndarray:
    """A function to compute the matrix of differences X[i] - X[j] of a vector X.
    
    Parameter
    ---------
    - X: np.array
        The vector for which we compute the difference matrix.

    Returns
    -------
    - D: np.ndarray
        The matrix of differences X[i] - X[j] of the vector X.
    """
    n = len(X)
    
    D = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            D[i][j] = X[i] - X[j]
            D[j][i] = X[j] - X[i]
    
    return D