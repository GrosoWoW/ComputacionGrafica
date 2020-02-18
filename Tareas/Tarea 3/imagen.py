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
    

    # Grafico
    fig, ax = mpl.subplots(1,1)
    pcm = ax.pcolormesh(Z, cmap='RdBu_r')
    fig.colorbar(pcm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Solución de la ecuación de Laplace.\n Con condicion de Neumann en el crater del volcan.')
    ax.set_aspect('equal', 'datalim')

    #ax.imshow(ub.T)
    mpl.show()

primero()
