# coding=utf-8


import numpy as np
import matplotlib.pyplot as mpl

def primero():
    # Definicion de variables
    H = 10000
    W = 20000

    # Defino las condiciones del Volcan
    ARRIBA = 0
    ABAJO = 0
    IZQUIERDA = 0
    DERECHA = 0

    # Discretizacion del volcan
    # 
    # Izquierda Derecha Abajo y Arriba son conocidos (Dirichlet condition)
    # Crater es desconocido (Neumann condition)



    # Variables
    Z = np.load("solution.npy")
    X = np.arange(0, len(Z)*2 - 1 , 1)
    Y = np.arange(0, len(Z) , 1)
    X, Y = np.meshgrid(X, Y)

    # Curvas de nivel
    ache = 20000/(len(Z)*2)

    fig, ax = mpl.subplots(1,1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Solución de la ecuación de Laplace.\n Con condicion de Neumann en el crater del volcan con curvas de nivel.')
    ax.set_aspect('equal', 'datalim')
    mpl.contour(ache*X, ache*Y, Z)
    mpl.xlabel('x')
    mpl.ylabel('y')

    #ax.imshow(ub.T)
    mpl.show()


primero()