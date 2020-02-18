# coding=utf-8


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as mpl
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

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


    #print(nh, nv)

    # Defino el dominio de los puntos
  

    # Variables
    Z = np.load("solution.npy")
    X = np.arange(0, len(Z)*2 - 1 , 1)
    Y = np.arange(0, len(Z) , 1)
    X, Y = np.meshgrid(X, Y)

    ache = 20000/(len(Z)*2)
    # Grafico
    fig = mpl.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Solución de la ecuación de Laplace.\n Con condicion de Neumann en el crater del volcan como superficie.')
    ax.set_aspect('equal', 'datalim')
 

    # Plot the surface.
    surf = ax.plot_surface(ache*X, ache*Y, ache*Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)


    mpl.show()



primero()