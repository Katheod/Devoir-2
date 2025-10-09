import numpy as np
def validerCholeski(matrix):
    symetrie = True
    det_positif = False
    n = len(matrix)

    for i in range(n):
        for j in range(n):
            if matrix[i][j] != matrix[j][i]:
                symetrie = False
                break

    if np.linalg.det(matrix) > 0:
        det_positif = True

    return (symetrie and det_positif)


def Choleski(matrix):
    assert validerCholeski(matrix), "La matrice n'est pas symétrique définie positive"

    res = np.zeros_like(matrix)
    
    l11 = np.sqrt(matrix[0][0])
    res[0][0] = l11
    n = len(matrix)

    for i in range(1, n):
        li1 = matrix[i][0]/ l11
        res[i][0] = li1
    
    for k in range(1, n):
        sum1 = 0
        for j in range(k):
            sum1 += res[k][j]**2
        lkk = np.sqrt(matrix[k][k] - sum1)
        res[k][k] = lkk

        for i in range(k+1, n):
            sum2 = 0
            for j in range(k):
                sum2 += res[i][j]*res[k][j]
            lik = (matrix[i][k] - sum2)/lkk
            res[i][k] = lik
    
    return res