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
    LT = np.matrix.transpose(L)

    # Résoudre Ly = b.
    y = np.linalg.solve(L, b)
    # Résoudre LT*x = y.
    x = np.linalg.solve(LT, y)

    return x


# Le dtype=float est essentielle à la résolution du système.
# Sans cela, numpy convertit les valeurs en int et la racine carrée d'un int n'est pas un int. 
# Ce qui donne un résultat erroné.
A = np.array([[10, 7, 8, 7],
              [7, 5, 6, 5],
              [8, 6, 10, 9],
              [7, 5, 9, 10]], dtype=float)

b = np.array([32, 23, 33, 31], dtype=float)

print(f"Le vecteur x = {ResolutionCholeski(A, b)}")

#Inverse de A


e1 = np.array([1,0,0,0])
e2 = np.array([0,1,0,0])
e3 = np.array([0,0,1,0])
e4 = np.array([0,0,0,1])

c1 = ResolutionCholeski(A,e1)
c2 = ResolutionCholeski(A,e2)
c3 = ResolutionCholeski(A,e3)
c4 = ResolutionCholeski(A,e4)


A_inv = np.array([c1,c2,c3,c4]).T

print(f"L'inverse de la matrice A")
print(A_inv)


#Normes l1, l2, infinie
for i in range(3):
    if i==0:
        s=1
        nom = "l1"
    elif i==1: 
        s=2
        nom = "l2"
    elif i==2:
        s=np.inf
        nom = "l_infini"

    r = np.array([0.000007, 0, 0.00002,0])
    norme_A = np.linalg.norm(A, s)
    norme_A_inv = np.linalg.norm(A_inv, s)
    norme_r = np.linalg.norm(r,s)
    norme_b = np.linalg.norm(b,s)

    erreur_relative_b = norme_r/norme_b
    cond_A = norme_A * norme_A_inv

    borne_inf_erreur_x = 1/cond_A * erreur_relative_b
    borne_sup_erreur_x = cond_A * erreur_relative_b

    print(f"\nNorme {nom}:")
    print(f"||A|| = {norme_A:.2f}")
    print(f"||A^(-1)|| = {norme_A_inv:.2f}")
    print(f"cond(A) = {cond_A:.2f}")
    print(f"||b - b*|| = {norme_r:.2E}")
    print(f"||b*|| = {norme_b:.2f}")
    print(f"||b - b*|| / ||b*|| = {erreur_relative_b:.2E}")
    print(f"Intervalle: [{borne_inf_erreur_x:.2E} ; {borne_sup_erreur_x:.2E}]")
