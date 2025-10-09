from Choleski import Choleski
import numpy as np

def ResolutionCholeski(A, b):
    L = Choleski(A)
    LT = np.matrix_transpose(L)

    y = np.linalg.solve(L, b)
    x = np.linalg.solve(LT, y)

    return x



A = np.array([[10, 7, 8, 7],
              [7, 5, 6, 5],
              [8, 6, 10, 9],
              [7, 5, 9, 10]])

b = np.array([32, 23, 33, 31])

(ResolutionCholeski(A, b))