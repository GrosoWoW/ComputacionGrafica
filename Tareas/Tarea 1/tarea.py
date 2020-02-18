# coding=utf-8

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import transformations as tr
import sys
import random

# We will use 32 bits data, so an integer has 4 bytes
# 1 byte = 8 bits
INT_BYTES = 4

# Transformation states that will operate over the shape
TR_STANDARD      = 0
TR_ROTATE_ZP     = 1
TR_ROTATE_ZM     = 2
TR_TRANSLATE     = 3
TR_UNIFORM_SCALE = 4
TR_NONUNIF_SCALE = 5
TR_REFLEX_Y      = 6
TR_SHEARING_XY   = 7

# Shapes
SP_FISH1      = 0
SP_FISH2      = 1
SP_FISH3      = 2
SP_TERRENO    = 3
SP_AGUA       = 4
SP_TERRENOSUB = 5
SP_BURBUJA    = 6
SP_COFRE      = 7
SP_ALGAS      = 8

# A class to store the application control
class Controller:

    x = 0.0
    y = 0.0
    theta = 0.0
    rotate = True
    showTransform = TR_STANDARD
    fillPolygon = True
    animated = False
    burbujas = False
    pescado = False
    def __init__(self):

        self.leftClickOn = False
        self.mousePos = (0.0, 0.0)


# we will use the global controller as communication with the callback function
controller = Controller()

def cursor_pos_callback(window, x, y):
    global controller
    controller.mousePos = (x,y)


def mouse_button_callback(window, button, action, mods):

    global controller

    """
    glfw.MOUSE_BUTTON_1: left click
    glfw.MOUSE_BUTTON_2: right click
    glfw.MOUSE_BUTTON_3: scroll click
    """

    if (action == glfw.PRESS or action == glfw.REPEAT):
        if (button == glfw.MOUSE_BUTTON_1):
            controller.leftClickOn = True
            print("Mouse click - button 1")

        if (button == glfw.MOUSE_BUTTON_2):
            print("Mouse click - button 2:", glfw.get_cursor_pos(window))

        if (button == glfw.MOUSE_BUTTON_3):
            print("Mouse click - button 3")

    elif (action ==glfw.RELEASE):
        if (button == glfw.MOUSE_BUTTON_1):
            controller.leftClickOn = False

def scroll_callback(window, x, y):

    print("Mouse scroll:", x, y)

def getTransform(showTransform, theta):

    if showTransform == TR_STANDARD:
        return tr.identity()

    elif showTransform == TR_ROTATE_ZP:
        return tr.rotationZ(theta)

    elif showTransform == TR_ROTATE_ZM:
        return tr.rotationZ(-theta)

    elif showTransform == TR_TRANSLATE:
        return tr.translate(0.3 * np.cos(theta), 0.3 * np.cos(theta), 0)

    elif showTransform == TR_UNIFORM_SCALE:
        return tr.uniformScale(0.7 + 0.5 * np.cos(theta))

    elif showTransform == TR_NONUNIF_SCALE:
        return tr.scale(
            1.0 - 0.5 * np.cos(1.5 * theta),
            1.0 + 0.5 * np.cos(2 * theta),
            1.0)

    elif showTransform == TR_REFLEX_Y:
        return tr.scale(1,-1,1)

    elif showTransform == TR_SHEARING_XY:
        return tr.shearing(0.3 * np.cos(theta), 0, 0, 0, 0, 0)
    
    else:
        # This should NEVER happend
        raise Exception()


def on_key(window, key, scancode, action, mods):

    if action != glfw.PRESS:
        return
    
    global controller

    if key == glfw.KEY_A:
        print('Toggling animated')
        controller.animated = not controller.animated

    elif key == glfw.KEY_S:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_ESCAPE:
        sys.exit()

    elif key == glfw.KEY_SPACE:
        controller.pescado = not controller.pescado

    else:
        print('Unknown key. Try small numbers!')


# A simple class container to reference a shape on GPU memory
class GPUShape:
    vao = 0
    vbo = 0
    ebo = 0
    size = 0

def drawShape(shaderProgram, shape, transform):

    # Binding the proper buffers
    glBindVertexArray(shape.vao)
    glBindBuffer(GL_ARRAY_BUFFER, shape.vbo)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, shape.ebo)

    # updating the new transform attribute
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "transform"), 1, GL_FALSE, transform)

    # Describing how the data is stored in the VBO
    position = glGetAttribLocation(shaderProgram, "position")
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)
    
    color = glGetAttribLocation(shaderProgram, "color")
    glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    glEnableVertexAttribArray(color)

    # This line tells the active shader program to render the active element buffer with the given size
    glDrawElements(GL_TRIANGLES, shape.size, GL_UNSIGNED_INT, None)



def createFish():

    x = 0
    y = 0


    # Here the new shape will be stored
    gpuShape = GPUShape()

    # Defining locations and colors for each vertex of the shape
    
    vertexData = np.array([
    #   positions        colors
        
        x, y, 0.0, 1.0, 0.13, 0.003,           
        x, y-0.05, 0.0,  1.0, 0.13, 0.003,
        x+0.05,  y-0.025, 0.0, 1.0, 0.13, 0.003,
        x+0.1, y+0.03, 0.0, 1.0, 0.13, 0.003,
        x+0.1, y-0.08, 0.0, 1.0, 0.13, 0.003, 
        x+0.2, y-0.025, 0.0, 1.0, 1.0, 1.0,
        x+0.28, y, 0.0, 1.0, 0.13 ,0.003,
        x+0.25, y+0.05, 0.0, 1.0, 0.13, 0.003,
        x+0.28, y-0.05, 0.0, 1.0, 0.13, 0.003,
        x+0.25, y-0.085, 0.0, 1.0, 0.13, 0.003,
        x+0.15, y-0.025, 0.0, 1.0, 0.13, 0.003, 
        x+0.16, y+0.05, 0.0, 1.0, 1.0, 1.0,
        x+0.16, y-0.1, 0.0, 1.0, 1.0, 1.0,
        x+0.09,  y-0.025, 0.0, 1.0, 1.0, 1.0,
        x+0.1, y+0.03, 0.0, 1.0, 1.0, 1.0,
        x+0.1, y-0.08, 0.0, 1.0, 1.0, 1.0,
        x+0.02, y, 0.0, 0.0, 0.0, 0.0,
        x+0.02, y-0.02, 0.0, 0.0, 0.0, 0.0,
        x+0.04, y, 0.0, 0.0, 0.0, 0.0,
        x+0.04, y-0.02, 0.0, 0.0, 0.0, 0.0

    # It is important to use 32 bits data
        ], dtype = np.float32)

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = np.array([        
        19, 18, 16,
        16, 17, 19,
        13, 14, 15,
        0, 1, 2,
        0, 3, 2,
        1, 2, 4,
        4, 2, 3,
        4, 3, 5,
        10, 6, 7,
        10, 8, 9,
        11, 3, 2,
        12, 4, 2], dtype= np.uint32)

    gpuShape.size = len(indices)

    # VAO, VBO and EBO and  for the shape
    gpuShape.vao = glGenVertexArrays(1)
    gpuShape.vbo = glGenBuffers(1)
    gpuShape.ebo = glGenBuffers(1)

    # Vertex data must be attached to a Vertex Buffer Object (VBO)
    glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertexData) * INT_BYTES, vertexData, GL_STATIC_DRAW)

    # Connections among vertices are stored in the Elements Buffer Object (EBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices) * INT_BYTES, indices, GL_STATIC_DRAW)

    return gpuShape

def createFish1():

    x = 0
    y = 0

    # Here the new shape will be stored
    gpuShape = GPUShape()

    # Defining locations and colors for each vertex of the shape
    
    vertexData = np.array([
    #   positions        colors
        
        x, y, 0.0, 0.1333, 0.443, 0.701,           
        x, y-0.05, 0.0,0.1333, 0.443, 0.701,     
        x+0.02,y-0.025, 0.0,0.1333, 0.443, 0.701,    
        x+0.1, y+0.03, 0.0, 0.1333, 0.443, 0.701,     
        x+0.1, y-0.08, 0.0, 0.1333, 0.443, 0.701,     
        x+0.2, y-0.025, 0.0,0.1333, 0.443, 0.701,   
        x+0.28, y, 0.0, 237/255, 255/255, 33/255,        
        x+0.25, y+0.05, 0.0, 237/255, 255/255, 33/255, 
        x+0.28, y-0.05, 0.0, 237/255, 255/255, 33/255,    
        x+0.25, y-0.085,0.0,  237/255, 255/255, 33/255,     
        x+0.15, y-0.025, 0.0,0.1333, 0.443, 0.701,     
        x+0.1, y+0.05, 0.0,0.0, 0.0, 0.0,     
        x+0.15, y-0.05, 0.0,237/255, 255/255, 33/255,     
        x+0.09,  y-0.025, 0.0,0.1333, 0.443, 0.701,    
        x+0.1, y+0.03, 0.0,0.1333, 0.443, 0.701,     
        x+0.1, y-0.08, 0.0,0.1333, 0.443, 0.701,    
        x+0.02, y, 0.0, 0.0, 0.0, 0.0,
        x+0.02, y-0.02, 0.0, 0.0, 0.0, 0.0,
        x+0.04, y, 0.0, 0.0, 0.0, 0.0,
        x+0.04, y-0.02, 0.0, 0.0, 0.0, 0.0

    # It is important to use 32 bits data
        ], dtype = np.float32)

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = np.array([
        2, 12, 4,
        11, 3, 5,        
        19, 18, 16,
        16, 17, 19,
        13, 14, 15,
        0, 1, 2,
        0, 3, 2,
        1, 2, 4,
        4, 2, 3,
        4, 3, 5,
        10, 6, 7,
        10, 8, 9,
        11, 3, 2
        ], dtype= np.uint32)

    gpuShape.size = len(indices)

    # VAO, VBO and EBO and  for the shape
    gpuShape.vao = glGenVertexArrays(1)
    gpuShape.vbo = glGenBuffers(1)
    gpuShape.ebo = glGenBuffers(1)

    # Vertex data must be attached to a Vertex Buffer Object (VBO)
    glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertexData) * INT_BYTES, vertexData, GL_STATIC_DRAW)

    # Connections among vertices are stored in the Elements Buffer Object (EBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices) * INT_BYTES, indices, GL_STATIC_DRAW)

    return gpuShape

def createFish2():

    x = 0
    y = 0

    # Here the new shape will be stored
    gpuShape = GPUShape()

    # Defining locations and colors for each vertex of the shape
    
    vertexData = np.array([
    #   positions        colors
        
        x, y, 0.0, 0.972, 0.952, 0.207,           
        x, y-0.05, 0.0,  0.972, 0.952, 0.207,   
        x+0.05,  y-0.025, 0.0, 0.972, 0.952, 0.207,   
        x+0.1, y+0.03, 0.0, 0.972, 0.952, 0.207,   
        x+0.1, y-0.08, 0.0, 0.972, 0.952, 0.207,   
        x+0.2, y-0.025, 0.0, 0.972, 0.952, 0.207,   #
        x+0.28, y, 0.0, 0.972, 0.952, 0.207,      
        x+0.25, y+0.05, 0.0, 0.972, 0.952, 0.207,   
        x+0.28, y-0.05, 0.0, 0.972, 0.952, 0.207,   
        x+0.25, y-0.085, 0.0, 0.972, 0.952, 0.207,   
        x+0.15, y-0.025, 0.0, 0.972, 0.952, 0.207,   
        x+0.16, y+0.05, 0.0, 0.972, 0.952, 0.207,   
        x+0.16, y-0.1, 0.0, 0.972, 0.952, 0.207,   
        x+0.09,  y-0.025, 0.0, 0.972, 0.952, 0.207,   
        x+0.1, y+0.03, 0.0, 0.972, 0.952, 0.207,   
        x+0.1, y-0.08, 0.0, 0.972, 0.952, 0.207,   
        x+0.02, y, 0.0, 0.0, 0.0, 0.0,
        x+0.02, y-0.02, 0.0, 0.0, 0.0, 0.0,
        x+0.04, y, 0.0, 0.0, 0.0, 0.0,
        x+0.04, y-0.02, 0.0, 0.0, 0.0, 0.0,
        x+0.13, y-0.05, 0.0, 37/255, 40/255, 80/255,
        x+0.15, y-0.041, 0.0, 37/255, 40/255, 80/255,
        x+0.13, y-0.02, 0.0,37/255, 40/255, 80/255,
        x+0.151, y, 0.0, 37/255, 40/255, 80/255


    # It is important to use 32 bits data
        ], dtype = np.float32)

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = np.array([
        20, 22, 23,
        23, 21, 20,        
        19, 18, 16,
        16, 17, 19,
        13, 14, 15,
        0, 1, 2,
        0, 3, 2,
        1, 2, 4,
        4, 2, 3,
        4, 3, 5,
        10, 6, 7,
        10, 8, 9,
        11, 3, 2,
        12, 4, 2], dtype= np.uint32)

    gpuShape.size = len(indices)

    # VAO, VBO and EBO and  for the shape
    gpuShape.vao = glGenVertexArrays(1)
    gpuShape.vbo = glGenBuffers(1)
    gpuShape.ebo = glGenBuffers(1)

    # Vertex data must be attached to a Vertex Buffer Object (VBO)
    glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertexData) * INT_BYTES, vertexData, GL_STATIC_DRAW)

    # Connections among vertices are stored in the Elements Buffer Object (EBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices) * INT_BYTES, indices, GL_STATIC_DRAW)

    return gpuShape

n = 0
listaPeces = []
transformaciones = []

class Pescado():
    def __init__(self):
        self.x = None
        self.y = None
        self.rotacion = []
        self.azar = None
        self.mov = None
        self.tipo = None
    
def createAgua():

    # Here the new shape will be stored
    gpuShape = GPUShape()

    # Defining locations and colors for each vertex of the shape
    
    vertexData = np.array([
    #   positions        colors
        -1.0, -1.0, 0.0,  0.04, 0.74, 0.94,
         1.0, -1.0, 0.0,  0.04, 0.74, 0.94,
         1.0,  1.0, 0.0,  0.0, 0.0, 1.0,
        -1.0,  1.0, 0.0,  0.0, 0.0, 1.0
    # It is important to use 32 bits data
        ], dtype = np.float32)

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = np.array(
        [0, 1, 2,
         2, 3, 0], dtype= np.uint32)

    gpuShape.size = len(indices)

    # VAO, VBO and EBO and  for the shape
    gpuShape.vao = glGenVertexArrays(1)
    gpuShape.vbo = glGenBuffers(1)
    gpuShape.ebo = glGenBuffers(1)

    # Vertex data must be attached to a Vertex Buffer Object (VBO)
    glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertexData) * INT_BYTES, vertexData, GL_STATIC_DRAW)

    # Connections among vertices are stored in the Elements Buffer Object (EBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices) * INT_BYTES, indices, GL_STATIC_DRAW)

    return gpuShape

def createTerreno():

    # Here the new shape will be stored
    gpuShape = GPUShape()

    # Defining locations and colors for each vertex of the shape
    
    vertexData = np.array([
    #   positions        colors
        -1.0, -1.0, 0.0,  0.45, 0.25, 0.05,
         1.0, -1.0, 0.0,  0.45, 0.25, 0.05,
         1.0,  -0.75, 0.0,  0.45, 0.25, 0.05,
        -1.0,  -0.75, 0.0,  0.45, 0.25, 0.05,  #Base
    # It is important to use 32 bits data
        ], dtype = np.float32)

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = np.array(
        [0, 1, 2,
         2, 3, 0
            ], dtype= np.uint32)

    gpuShape.size = len(indices)

    # VAO, VBO and EBO and  for the shape
    gpuShape.vao = glGenVertexArrays(1)
    gpuShape.vbo = glGenBuffers(1)
    gpuShape.ebo = glGenBuffers(1)

    # Vertex data must be attached to a Vertex Buffer Object (VBO)
    glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertexData) * INT_BYTES, vertexData, GL_STATIC_DRAW)

    # Connections among vertices are stored in the Elements Buffer Object (EBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices) * INT_BYTES, indices, GL_STATIC_DRAW)

    return gpuShape

def createTerrenoSub():

    # Here the new shape will be stored
    gpuShape = GPUShape()

    # Defining locations and colors for each vertex of the shape
    
    vertexData = np.array([
    #   positions        colors
        -1.0, -0.85, 0.0,  0.84, 0.57, 0.26,
         -0.5, -0.88, 0.0,  0.84, 0.57, 0.26,
         -0.2,  -0.65, 0.0,  0.84, 0.57, 0.26,
         0.5,  -0.65, 0.0, 0.84, 0.57, 0.26,
         0.8,  -0.7, 0.0, 0.84, 0.57, 0.26,
         1.0, -0.8, 0.0, 0.84, 0.57, 0.26
    # It is important to use 32 bits data
        ], dtype = np.float32)

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = np.array(
        [0, 1, 2,
        1, 2, 3,
        3, 4, 1,
        4, 5, 1
            ], dtype= np.uint32)

    gpuShape.size = len(indices)

    # VAO, VBO and EBO and  for the shape
    gpuShape.vao = glGenVertexArrays(1)
    gpuShape.vbo = glGenBuffers(1)
    gpuShape.ebo = glGenBuffers(1)

    # Vertex data must be attached to a Vertex Buffer Object (VBO)
    glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertexData) * INT_BYTES, vertexData, GL_STATIC_DRAW)

    # Connections among vertices are stored in the Elements Buffer Object (EBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices) * INT_BYTES, indices, GL_STATIC_DRAW)

    return gpuShape

def createBurbuja():

    # Here the new shape will be stored
    gpuShape = GPUShape()

    data = [0,0,0,1.0,1.0,1.0]
    polar = []

    for i in range(361):
        j = (i/180)*np.pi
        data = data +[0.05*np.cos(j), 0.05*np.sin(j),0,0,0,1]
        polar = polar + [0,i,i+1]

    vertexData = np.array(data,dtype =np.float32)

    indices = np.array(polar, dtype= np.uint32)
        
    gpuShape.size = len(indices)

    # VAO, VBO and EBO and  for the shape
    gpuShape.vao = glGenVertexArrays(1)
    gpuShape.vbo = glGenBuffers(1)
    gpuShape.ebo = glGenBuffers(1)

    glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertexData) * INT_BYTES, vertexData, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices) * INT_BYTES, indices, GL_STATIC_DRAW)

    return gpuShape

def createCofre(x,y):

    x3 = x
    y3 = y

    # Here the new shape will be stored
    gpuShape = GPUShape()

    # Defining locations and colors for each vertex of the shape
    
    vertexData = np.array([
    #   positions        colors
        x3, y3, 0.0,  0.46, 0.23, 0.15,
        x3+0.3, y3, 0.0,  0.46, 0.23, 0.15,
        x3+0.3,  y3+0.15, 0.0,  0.46, 0.23, 0.15,
        x3, y3+0.15, 0.0, 0.46, 0.23, 0.15,  #Base
        x3, y3+0.2, 0.0, 1.0, 1.0, 1.0,
        x3+0.3,y3+0.2, 0.0, 1.0, 1.0, 1.0,
        x3+0.12, y3+0.18, 0.0, 0.98, 0.82, 0.003,
        x3+0.18, y3+0.12, 0.0, 0.98, 0.82, 0.003,
        x3+0.12, y3+0.12, 0.0, 0.98, 0.82, 0.003,
        x3+0.18, y3+0.18, 0.0, 0.98, 0.82, 0.003

    # It is important to use 32 bits data
        ], dtype = np.float32)

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = np.array(
        [6, 7, 9,
         6, 8, 7,
         0, 1, 2,
         2, 3, 0,
         3, 4, 5,
         5, 2, 3

            ], dtype= np.uint32)

    gpuShape.size = len(indices)

    # VAO, VBO and EBO and  for the shape
    gpuShape.vao = glGenVertexArrays(1)
    gpuShape.vbo = glGenBuffers(1)
    gpuShape.ebo = glGenBuffers(1)

    # Vertex data must be attached to a Vertex Buffer Object (VBO)
    glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertexData) * INT_BYTES, vertexData, GL_STATIC_DRAW)

    # Connections among vertices are stored in the Elements Buffer Object (EBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices) * INT_BYTES, indices, GL_STATIC_DRAW)

    return gpuShape

def createAlgas():

    x = 0
    y = 0

    # Here the new shape will be stored
    gpuShape = GPUShape()

    # Defining locations and colors for each vertex of the shape
    
    vertexData = np.array([
    #   positions        colors
        x, y, 0.0, 0.0, 1.0, 0.0,
        x+0.03, y, 0.0,  0.0, 1.0, 0.0,
        x,  y+0.3, 0.0,  0.0, 1.0, 0.0,
        x+0.03,y+0.3, 0.0, 0.0, 1.0, 0.0

    # It is important to use 32 bits data
        ], dtype = np.float32)

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = np.array(
        [0, 1, 2,
         2, 3, 1], dtype= np.uint32)

    gpuShape.size = len(indices)

    # VAO, VBO and EBO and  for the shape
    gpuShape.vao = glGenVertexArrays(1)
    gpuShape.vbo = glGenBuffers(1)
    gpuShape.ebo = glGenBuffers(1)

    # Vertex data must be attached to a Vertex Buffer Object (VBO)
    glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
    glBufferData(GL_ARRAY_BUFFER, len(vertexData) * INT_BYTES, vertexData, GL_STATIC_DRAW)

    # Connections among vertices are stored in the Elements Buffer Object (EBO)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices) * INT_BYTES, indices, GL_STATIC_DRAW)

    return gpuShape

if __name__ == "__main__":


    # Initialize glfw
    if not glfw.init():
        sys.exit()

    width = 600
    height = 600

    window = glfw.create_window(width, height, "Peces en el mar", None, None)

    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, scroll_callback)

    # Defining shaders for our pipeline
    vertex_shader = """
    #version 130
    in vec3 position;
    in vec3 color;

    out vec3 fragColor;

    uniform mat4 transform;

    void main()
    {
        fragColor = color;
        gl_Position = transform * vec4(position, 1.0f);
    }
    """

    fragment_shader = """
    #version 130

    in vec3 fragColor;
    out vec4 outColor;

    void main()
    {
        outColor = vec4(fragColor, 1.0f);
    }
    """

    # Assembling the shader program (pipeline) with both shaders
    shaderProgram = OpenGL.GL.shaders.compileProgram(
        OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
        OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    # Telling OpenGL to use our shader program
    glUseProgram(shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.15, 0.15, 0.15, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    gpuAgua = createAgua()
    gpuTerreno = createTerreno()
    gpuTerrenoSub = createTerrenoSub()
    gpuBurbuja = createBurbuja()
    gpuFish = createFish()
    gpuFish1 = createFish1()
    gpuFish2 = createFish2()
    gpuCofre1 = createCofre(0.65,-0.80)
    gpuCofre2 = createCofre(-0.75,-0.85)
    gpuAlgas = createAlgas()

    t0 = glfw.get_time()

    while not glfw.window_should_close(window):

        t1 = glfw.get_time()
        dt= t1-t0
        t0 = t1

        # Using GLFW to check for input events
        glfw.poll_events()

        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if controller.animated:
            # Using the time as the theta parameter
            theta = glfw.get_time()
        else:
            theta = np.pi / 6
        
        transform = getTransform(controller.showTransform, theta)

        mousePosX = 2 * (controller.mousePos[0] - width/2) / width
        mousePosY = 2 * (height/2 - controller.mousePos[1]) / height

        if not controller.pescado:
            if n < 1:
                
                x4 = random.uniform(-1,1)
                y4 = random.uniform(-1,1)
                g = random.uniform(-1,1)
                pez = Pescado()
                pez.x = x4
                pez.y = y4
                pez.azar = g
                pez.mov = 1
                pez1 = Pescado()
                pez1.x = x4
                pez1.y = y4
                pez1.azar = g
                pez1.mov = 2
                pez2 = Pescado()
                pez2.x = x4
                pez2.y = y4
                pez2.azar = g
                pez2.mov = 3
                pez.tipo = gpuFish
                pez1.tipo = gpuFish1
                pez2.tipo = gpuFish2
                listaPeces += [pez]
                listaPeces += [pez1]
                listaPeces += [pez2]
                n += 1
                controller.pescado = not controller.pescado

            else :

                x4 = random.uniform(-1,1)
                y4 = random.uniform(-1,1)
                k = random.randrange(1,4)
                h = random.randint(1,3)
                print (h)
                pez = Pescado()
                pez.x = x4
                pez.y = y4
                pez.azar = x4
                pez.mov = h
                controller.pescado = not controller.pescado

                if k ==1:
                    pez.tipo = gpuFish
                    listaPeces += [pez]

                elif k ==2:
                    pez.tipo = gpuFish1
                    listaPeces += [pez]
                  
                elif k ==3:
                    pez.tipo = gpuFish2
                    listaPeces += [pez]
    
        def Movimiento (x,y):
            if x ==1:
                return tr.matmul([tr.translate(0,0,0),tr.translate(y*np.cos(t1*y)
            *np.sin(t1),y*np.cos(t1),0), tr.shearing(np.sin(t1)/50,np.cos(t1)/8,0,0,0,0)])

            elif x ==2 :
                return tr.matmul([tr.translate(np.cos(t1*y),
            y*np.sin(t1*y)*np.sin(t1*y),0),tr.uniformScale(0.5), tr.shearing(np.sin(t1)/8,np.cos(t1)/8,0,0,0,0)])

            elif x ==3 :
                return tr.matmul([tr.translate(y*np.sin(t1*y),np.cos(t1*y),0), tr.uniformScale(0.8), tr.shearing(np.sin(t1)/8,np.cos(t1)/8,0,0,0,0)])

            elif x == 4:
                return tr.matmul([tr.translate(1000000000000,100000000000000,0)])

        for i in range (len(listaPeces)):

            
            listaPeces[i-1].rotacion = Movimiento(listaPeces[i-1].mov,listaPeces[i-1].azar)  #Pez 1
            if listaPeces[i-1].rotacion[3][0] > 0.97 or listaPeces[i-1].rotacion[3][1] > 0.97:
                listaPeces[i-1].rotacion = tr.matmul([tr.translate(10000000000000000,100000000000000000000,0)])
                listaPeces[i-1].mov = 4

            if controller.leftClickOn:
                if listaPeces[i-1].rotacion[3][0] < mousePosX and listaPeces[i-1].rotacion[3][0]+0.13 > mousePosX and listaPeces[i-1].rotacion[3][1]-0.1 < mousePosY and listaPeces[i-1].rotacion[3][1]+0.1 > mousePosY:
                    listaPeces[i-1].rotacion = tr.matmul([tr.translate(10000000000000000,100000000000000000000,0)])
                    listaPeces[i-1].mov = 4
                    controller.leftClickOn = not controller.leftClickOn


            transformFish = listaPeces[i-1].rotacion 
            drawShape(shaderProgram, listaPeces[i-1].tipo, transformFish)
        
        transformAlgas = tr.matmul([tr.translate(0,-0.75,0), tr.shearing(0,np.sin(t1)/2,0,0,0,0)])
        drawShape(shaderProgram, gpuAlgas, transformAlgas) 

        transformAlgas = tr.matmul([tr.translate(0.05,-0.75,0), tr.shearing(0,np.sin(t1)/4,0,0,0,0)])
        drawShape(shaderProgram, gpuAlgas, transformAlgas)

        transformAlgas = tr.matmul([tr.translate(0.10,-0.75,0), tr.shearing(0,np.sin(t1)/8,0,0,0,0)])
        drawShape(shaderProgram, gpuAlgas, transformAlgas)

        transformAlgas = tr.matmul([tr.translate(0.15,-0.75,0), tr.shearing(0,np.sin(t1)/16,0,0,0,0)])
        drawShape(shaderProgram, gpuAlgas, transformAlgas)

        transformAlgas = tr.matmul([tr.translate(0.20,-0.75,0), tr.shearing(0,np.sin(t1)/32,0,0,0,0)])
        drawShape(shaderProgram, gpuAlgas, transformAlgas)

        transformAlgas = tr.matmul([tr.translate(0.25,-0.75,0), tr.shearing(0,np.sin(t1)/64,0,0,0,0)])
        drawShape(shaderProgram, gpuAlgas, transformAlgas)

        transformCofre1 = getTransform(controller.showTransform,0)
        drawShape(shaderProgram, gpuCofre1, transformCofre1)

        transformCofre2 = getTransform(controller.showTransform,0)
        drawShape(shaderProgram, gpuCofre2, transformCofre2 )

        theta = np.tan(glfw.get_time())

        transformBurbuja1 = tr.matmul([tr.uniformScale(0.5),tr.translate(-0.5, 0.3+theta, 0)])   # --------- Burbujas---------
        drawShape(shaderProgram, gpuBurbuja, transformBurbuja1)
        transformBurbuja2 = tr.matmul([tr.uniformScale(0.5),tr.translate(-0.7, 0.8+theta, 0)])
        drawShape(shaderProgram, gpuBurbuja, transformBurbuja2)
        transformBurbuja3 = tr.matmul([tr.uniformScale(0.5),tr.translate(0.5, 0.9+theta, 0)])
        drawShape(shaderProgram, gpuBurbuja, transformBurbuja3)
        transformBurbuja4 = tr.matmul([tr.uniformScale(0.5),tr.translate(0.2, 0.4+theta, 0)])
        drawShape(shaderProgram, gpuBurbuja, transformBurbuja4)
        transformBurbuja5 = tr.matmul([tr.uniformScale(0.5),tr.translate(-0.5, -0.3+theta, 0)])
        drawShape(shaderProgram, gpuBurbuja, transformBurbuja5)
        transformBurbuja6 = tr.matmul([tr.uniformScale(0.5),tr.translate(0.8, -0.3+theta, 0)])
        drawShape(shaderProgram, gpuBurbuja, transformBurbuja6)
        transformBurbuja7 = tr.matmul([tr.uniformScale(0.5),tr.translate(0.1, -0.65+theta, 0)])
        drawShape(shaderProgram, gpuBurbuja, transformBurbuja7)                                     #----------------Burbujas----------


        transformTerrenoSub = getTransform(controller.showTransform, 0)  #Terreno
        drawShape(shaderProgram, gpuTerrenoSub, transformTerrenoSub)
        
        transformTerreno = getTransform(controller.showTransform, 0)  #Terreno
        drawShape(shaderProgram, gpuTerreno, transformTerreno)


        transformAgua = getTransform(controller.showTransform, 0)    #Agua
        drawShape(shaderProgram, gpuAgua, transformAgua )


        # Once the drawing is rendered, buffers are swap so an uncomplete drawing is never seen.
        glfw.swap_buffers(window)

    glfw.terminate()