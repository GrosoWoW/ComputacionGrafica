# coding=utf-8

import numpy as np
import matplotlib.pyplot as mpl
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix
import sys

def primero(x, E, cargar):
    # Definicion de variables
    H = 10000
    W = 20000
    h = x

    # Defino las condiciones del Volcan
    ARRIBA = 0
    ABAJO = 0
    IZQUIERDA = 0
    DERECHA = 0

    # Discretizacion del volcan
    # 
    # Izquierda Derecha Abajo y Arriba son conocidos (Dirichlet condition)
    # Crater es desconocido (Neumann condition)

    nh = int(W / h)
    nv = int(H / h) - 1


    # Defino el dominio de los puntos
    N = nh * nv

    vol1 = int(nh*225/1000)  # tamaño de el primer lado en x
    altura = 0

    for i in range (vol1):  # Calcula la altura
    
        altura = i
    def getK(i,j):
        return j * nh + i

    def getIJ(k):
        i = k % nh
        j = k // nh
        return (i, j)

    A = lil_matrix((N, N))
    b = np.zeros((N,))

    posc = (nh*41/100)
    posc1 = (nh*75/1000)

    inicio = int(nh*52/100) - 2  # donde inicia el volcan
    cra = int(nh*11/100)  # tamaño del volcan

    inicio1 = int(nh*63/100)-1   # inicio segundo lado
    cra1 = int(nh*220/1000) 

    prim = int(nh*295/1000)

    for i in range(0, nh):
        for j in range(0, nv):

            # We will write the equation associated with row k
            k = getK(i,j)

            # We obtain indices of the other coefficients
            k_up = getK(i, j+1)
            k_down = getK(i, j-1)
            k_IZQUIERDA = getK(i-1, j)
            k_DERECHA = getK(i+1, j)

            # Definicion de la ecuacion de cada punto dependiendo de su posicion

            # Crater

            if inicio <= i and i <= inicio + cra  and j == altura-1:

                A[k, k_up] = 1
                A[k, k_IZQUIERDA] = 1
                A[k, k_DERECHA] = 1
                A[k, k] = -4
                b[k] = -2*h*E

            # Lado 1 del volcan

            elif prim-1<= i and i<= prim+vol1-3 and i-j == prim-1:

                A[k, k_up] = 1
                A[k, k_IZQUIERDA] = 1
                A[k, k] = -4
                b[k] = 0

            # Lado 2 del volcan

            elif int (inicio1) <= i and i <= inicio1 + cra1/2 and i-j == inicio1 + (cra1/2) - altura + cra1:

                A[k, k_up] = 1
                A[k, k_DERECHA] = 1
                A[k, k] = -4
                b[k] = 0


            elif 1 <= i and i <= nh - 2 and 1 <= j and j <= nv - 2:
                A[k, k_up] = 1
                A[k, k_down] = 1
                A[k, k_IZQUIERDA] = 1
                A[k, k_DERECHA] = 1
                A[k, k] = -4
                b[k] = 0
            
            # IZQUIERDA 
            elif i == 0 and 1 <= j and j <= nv - 2:
                A[k, k_up] = 1
                A[k, k_down] = 1
                A[k, k_DERECHA] = 1
                A[k, k] = -4
                b[k] = 0
            
            # DERECHA 
            elif i == nh - 1 and 1 <= j and j <= nv - 2:
                A[k, k_up] = 1
                A[k, k_down] = 1
                A[k, k_IZQUIERDA] = 1
                A[k, k] = -4
                b[k] = 0
            
            # ABAJO (dentro del volcan)
            elif posc <= i and i <= posc1  and j == 0:
                A[k, k_up] = 0
                A[k, k_IZQUIERDA] = 0
                A[k, k_DERECHA] = 0
                A[k, k] = -4
                b[k] = 0

            # ABAJO 
            elif 1 <= i and i <= nh - 2 and j == 0:
                A[k, k_up] = 1
                A[k, k_IZQUIERDA] = 1
                A[k, k_DERECHA] = 1
                A[k, k] = -4
                b[k] = 0
            
            # ARRIBA 
            elif 1 <= i and i <= nh - 2 and j == nv - 1:
                A[k, k_down] = 1
                A[k, k_IZQUIERDA] = 1
                A[k, k_DERECHA] = 1
                A[k, k] = -4
                b[k] = 0
            # corner lower IZQUIERDA
            elif (i, j) == (0, 0):
                A[k, k] = 1
                b[k] = 0

            # corner lower DERECHA
            elif (i, j) == (nh - 1, 0):
                A[k, k] = 1
                b[k] = 0

            # corner upper IZQUIERDA
            elif (i, j) == (0, nv - 1):
                A[k, k] = 1
                b[k] = 0

            # corner upper DERECHA
            elif (i, j) == (nh - 1, nv - 1):
                A[k, k] = 1
                b[k] = 0

            else:
                print("Point (" + str(i) + ", " + str(j) + ") missed!")
                print("Associated point index is " + str(k))
                raise Exception()


    # Resuelve el sistema
    A= A.tocsc() 
    x = spsolve(A, b)    
    u = np.zeros((nh,nv))

    cra = int(nh*11/100)
    for k in range(0, N):
        i,j = getIJ(k)
        u[i,j] = x[k]

    # Se añaden los resultados
    ub = np.zeros((nh + 1, nv + 2))
    ub[1:nh + 1, 1:nv + 1] = u[:,:]

    # Dirichlet condición

    # ARRIBA 
    ub[0:nh + 2, nv + 1] = ARRIBA

    # ABAJO PRIMERA PARTE
    prim = int(nh*295/1000)
    ub[0:prim + 2, 0] = ABAJO

    # ABAJO SEGUNDA PARTE
    sec = int(nh*705/1000)
    ub[sec: nh + 2, 0] = ABAJO

    # Lado 1 del volcan

    vol1 = int(nh*225/1000)

    for i in range(altura):

        ub[inicio-i:inicio+3+cra+int(i/2), altura-1-i] = 1
      
    for i in range (vol1):
        ub[prim + i ,i] = ABAJO
            
        altura = i

    # Lado 2 del volcan
    inicio1 = int(nh*63/100)-1
    cra1 = int(nh*220/1000) 

    #for i in range(0,cra1):

        #ub[int(inicio1 + i/2), altura - i] = ABAJO


    inicio = int(nh*52/100)
    cra = int(nh*11/100)
    ub[int(inicio) :int(inicio + cra) , altura] = 0

    # IZQUIERDA
    ub[0, 1:nv + 1] = IZQUIERDA

    # DERECHA
    ub[nh , 1:nv +1] = DERECHA

    np.save(cargar,ub.T)
    # Variables
    X = np.arange(0, nh + 1, 1)
    Y = np.arange(0, nv + 2, 1)
    X, Y = np.meshgrid(X, Y)
    Z = ub.T


if __name__ == "__main__":
    primero(int(sys.argv[1]) , int(sys.argv[2]), sys.argv[3])


