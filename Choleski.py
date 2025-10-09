import numpy as np

def Choleski(A):
    """Effectue la décomposition de Choleski si une matrice est symétrique et définie positive. Sinon, elle lève une exception.

    Args:
        A (np.ndarray): La matrice à décomposer.

    Returns:
        np.ndarray: La matrice triangulaire inférieure.
    """
    # Critères de validation.
    symetrie = True
    det_positif = False
    n = len(A)

    # Vérification de la symétrie.
    for i in range(n):
        for j in range(n):
            if A[i][j] != A[j][i]:
                symetrie = False
                break

    # Vérification du déterminant positif.
    if np.linalg.det(A) > 0:
        det_positif = True

    validation_critere = (symetrie and det_positif)



    # Validation de la matrice.
    assert validation_critere, "La matrice n'est pas symétrique définie positive"

    # Initialisation de la matrice résultat.
    res = np.zeros_like(A)

    # Calcul du premier pivot.
    l11 = np.sqrt(A[0][0])
    res[0][0] = l11

    n = len(A)

    # Calcul de la première colonne.
    for i in range(1, n):
        li1 = A[i][0]/ l11
        res[i][0] = li1
    
    # Calcul des termes diagonal (pivot).
    for k in range(1, n):
        sum1 = 0
        for j in range(k):
            sum1 += res[k][j]**2
        lkk = np.sqrt(A[k][k] - sum1)
        res[k][k] = lkk

        # Calcul des autres termes.
        for i in range(k+1, n):
            sum2 = 0
            for j in range(k):
                sum2 += res[i][j]*res[k][j]
            lik = (A[i][k] - sum2)/lkk
            res[i][k] = lik
    
    return res