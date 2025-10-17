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



#Inverse de A

e1 = np.matrix.transpose(np.array([1,0,0,0]))
e2 = np.matrix.transpose(np.array([0,1,0,0]))
e3 = np.matrix.transpose(np.array([0,0,1,0]))
e4 = np.matrix.transpose(np.array([0,0,0,1]))

c1 = ResolutionCholeski(A,e1)
c2 = ResolutionCholeski(A,e2)
c3 = ResolutionCholeski(A,e3)
c4 = ResolutionCholeski(A,e4)

A1 = np.array([[c1],[c2],[c3],[c4]])
