import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, LassoLarsCV
import cvxpy as cp
from random import random
from utils.lin_alg import half_vectorize, diff_matrix
from dataclasses import dataclass

@dataclass
class RegSolution:
    """A regression solution data class.

    Attributes
    ----------
    - sol: np.ndarray
        The regression solution, as an array.
    - res_norm: float
        The norm of the residual resulting from the solution.
    """
    sol: np.ndarray
    res_norm: float

class SignedDifferencesOLS:
    """The class for the solver of the Signed Differences OLS problem, based on a particular library.

    Attribute
    ---------
    - library: str
        The library to be used. (This is `cvxpy` by default as it yields the best results.)

    Methods
    -------
    - construct_design_matrix
        Constructs the design matrix for the equivalent vectorized problem. (See the original `Problem.md`.)
    - solve
        Solves the OLS problem via Non-Negative Least Squares from the selected library.
    """
    def __init__(self, library : str = 'cvxpy'):
        self.library = library

    # Credit to @arahimyar for this implementation of the construction of the design matrix
    def construct_design_matrix(self, n: int) -> np.ndarray:
        """A method to construct the design matrix for the equivalent vectorized problem.

        Parameter
        ---------
        - n: int
            The size of the (skew-symmetric) difference matrix.
        Returns
        -------
        - A: np.ndarray
            The design matrix A for the equivalent vectorized problem.
        """
        
        A = np.zeros((n * (n-1) // 2, n))
        
        count = 0
        
        for j in range(n-1):
            for i in range(n-j-1):
                A[count+i,j] = 1
                A[count+i, j+i+1] = -1
                
            count += n-1 - j
        
        return A
    
    def solve(self, data: np.ndarray) -> RegSolution:
        """
        A method for solving the Signed Differences OLS problem.

        Parameter
        ---------
        - data: np.ndarray
            The data, given as a skew-symmetric matrix of differences. (Or approximately skew-symmetric, if noise-contaminated.)

        Returns
        -------
        - solution: RegSolution
            The solution as a `RegSolution` dataclass containing the solution vector, and the norm of the residual given by this solution.
        """
        n = len(data)
        
        b = half_vectorize(data) 

        A = SignedDifferencesOLS.construct_design_matrix(self, n)
        
        if self.library == 'cvxpy':
            x = cp.Variable(n)
            
            cost = cp.sum_squares((A @ x) - b)
            
            prob = cp.Problem(
                cp.Minimize(cost),
                [x >= 0]
            )
            
            prob.solve()
            
            X_hat, res_norm = x.value, prob.solve()
        
        elif self.library == 'scipy':
            solution = nnls(A, b)
            
            X_hat, res_norm = solution[0], solution[1]
            
        elif self.library == 'sklearn':
            reg_nnls = LinearRegression(positive = True)
            
            reg_nnls.fit(A, b)
            
            X_hat = reg_nnls.coef_
            
            res_norm = np.linalg.norm(A @ X_hat - b)

        solution = RegSolution(sol=X_hat, res_norm=res_norm)
        
        return solution
