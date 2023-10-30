import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, LassoLarsCV
import cvxpy as cp
from random import random
from dataclasses import dataclass

@dataclass
class Solution:
    sol: np.ndarray
    res_norm: float

class SignedDifferencesOLS:
    def __init__(self, library = 'CVXPY'):
        self.library = library

    def generate_data(self, n, positive = True, noise = False):
        
        X = np.random.random(n) if positive else np.random.randn(n)
        
        D = diff_matrix(X) + 1e-5 * np.random.normal(n) if noise else diff_matrix(X)
        
        return {
            'nums' : X,
            'diff_matrix' : D
        }
    
    def construct_design_matrix(self, n):
        A = np.zeros((n * (n-1) // 2, n))
        
        count = 0
        
        for j in range(n-1):
            for i in range(n-j-1):
                A[count+i,j] = 1
                A[count+i, j+i+1] = -1
                
            count += n-1 - j
        
        return A
    
    def solve(self, data):
        n = len(data)
        
        b = half_vectorize(data) 

        A = SignedDifferencesOLS.construct_design_matrix(self, n)
        
        if self.library == 'CVXPY':
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
        
        return Solution(sol=X_hat, res_norm=res_norm)