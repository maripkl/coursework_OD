import numpy as np


def joint_diagonalization(I, D, return_diagonal=True):
    # T: T @ I @ T.T = eye, T @ D @ T.T = diagonal
    lmbda, L = np.linalg.eig(np.linalg.inv(I))
    Lmbda_sqrt = np.diag(lmbda ** 0.5)
    _, U = np.linalg.eig(Lmbda_sqrt @ L.T @ D @ L @ Lmbda_sqrt)
    T = U.T @ Lmbda_sqrt @ L.T
    if return_diagonal:
        return T, np.diag(T @ D @ T.T)
    else:
        return T