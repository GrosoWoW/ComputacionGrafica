import volcanic_eruption as t
import numpy as np
import matplotlib.pyplot as mpl
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix

def lector():

        condicion = True
        H = 10000
        W = 20000
        h = 500
        nh = int(W / h)
        nv = int(H / h) - 1

        E = 200
        print("Calculando...")
        while condicion:

                t.primero(h, E, "solution")
                matriz = np.load("solution.npy")
                condicionDesconocida = matriz[int(5000/h), int(7500/h)]
                
                if 2900000 <= condicionDesconocida and condicionDesconocida >= 3100000 :
                        condicion = False
                        print("Calculo Terminado")
                        print ("Valor: "+str(E))
                        break
                E += 200


lector()   
        
        

