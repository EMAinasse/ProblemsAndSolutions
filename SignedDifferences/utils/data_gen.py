""""Function to generate data."""

import numpy as np
from dataclasses import dataclass
from utils.lin_alg import diff_matrix

@dataclass
class SignedDifferencesData:
    nums: np.array
    diff_matrix: np.ndarray

def generate_data(n: int, positive: bool = True, noise: bool = False) -> SignedDifferencesData:
    """
    Parameters
    ----------
    - n: int
        The size of the vector of numbers from which the difference matrix is to be generated.
    - positive: bool
        A boolean to indicate whether or not the vector X of numbers should consist of positive numbers only (between 0 and 1).
    - noise: bool
        A boolean to indicate whether or not the difference matrix D should consist be contaminated by noise of the form 10^{-5} * Gaussian.
    """
    
    X = np.random.random(n) if positive else np.random.randn(n)
    
    D = diff_matrix(X) + 1e-5 * np.random.normal(n) if noise else diff_matrix(X)
    
    return SignedDifferencesData(nums = X, diff_matrix = D)