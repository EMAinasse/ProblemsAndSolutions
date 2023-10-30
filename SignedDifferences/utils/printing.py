"""A helper function to print the results."""

def results_message(X_hat, res_norm : float, lib: str) -> str:
    """This function prints a string displaying the results of the solution.
    
    Parameters
    ----------
    - X_hat:
        The solution to the Signed Differences problem.
    - res_norm: float
        The norm of the residual.
    - lib: str
        The library used to produce the solution.

    Returns
    -------
    - message
        A nice message displaying the results.
    """
    message = f"""
    The {lib}-based solution is:
    {X_hat}
    The norm of the residual is: {res_norm}.
    """
    return message

def vectors_comparison(X_hat, X, lib) -> str:
    """This function prints a message displaying both vectors and the max-norm of their difference.
    
    Parameters
    ----------
    - X_hat
        The solution to the Signed Differences OLS problem based on the matrix difference of X.
    - X
        The original vector from which the difference matrix was constructed.

    Returns
    -------
    - message: str
        The message displaying the comparison of the results.
    """
    max_norm = max(abs(X_hat - X))

    message = f"""
    The {lib}-based OLS solution (X_hat) is:
    {X_hat}
    The original vector (X) was:
    {X}
    The max-norm of their difference is: {max_norm}.
    """

    return message