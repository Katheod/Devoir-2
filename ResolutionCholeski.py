from Choleski import Choleski
import numpy as np

def ResolutionCholeski(A, b):
    """Résout un système d'équations linéaires Ax = b en utilisant la décomposition de Choleski.

    Args:
        A (np.ndarray): La matrice des coefficients.
        b (np.ndarray): Le vecteur des constantes.

    Returns:
        np.ndarray: La solution du système d'équations.
    """
    # Récupérer la décomposition de Choleski.
    L = Choleski(A)
    # Calculer la transposée de L.
    LT = np.matrix_transpose(L)

    # Résoudre Ly = b.
    y = np.linalg.solve(L, b)
    # Résoudre LT*x = y.
    x = np.linalg.solve(LT, y)

    return x



A = np.array([[10, 7, 8, 7],
              [7, 5, 6, 5],
              [8, 6, 10, 9],
              [7, 5, 9, 10]], dtype=float)

b = np.array([32, 23, 33, 31], dtype=float)

print(ResolutionCholeski(A, b))
