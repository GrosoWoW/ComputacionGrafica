
# coding=utf-8
"""
Daniel Calderon, CC3501, 2019-1
vertices and indices for simple shapes
"""
import numpy as np


# A simple class container to store vertices and indices that define a shape
class Shape:
    def __init__(self, vertices, indices, textureFileName=None):
        self.vertices = vertices
        self.indices = indices
        self.textureFileName = textureFileName


def createAxis(length=1.0):

    # Defining the location and colors of each vertex  of the shape
    vertices = [
    #    positions        colors
        -length,  0.0,  0.0, 0.0, 0.0, 0.0,
         length,  0.0,  0.0, 1.0, 0.0, 0.0,

         0.0, -length,  0.0, 0.0, 0.0, 0.0,
         0.0,  length,  0.0, 0.0, 1.0, 0.0,

         0.0,  0.0, -length, 0.0, 0.0, 0.0,
         0.0,  0.0,  length, 0.0, 0.0, 1.0]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
         0, 1,
         2, 3,
         4, 5]

    return Shape(vertices, indices)


def createRainbowTriangle():

    # Defining the location and colors of each vertex  of the shape
    vertices = [
    #   positions        colors
        -0.5, -0.5, 0.0,  1.0, 0.0, 0.0,
         0.5, -0.5, 0.0,  0.0, 1.0, 0.0,
         0.0,  0.5, 0.0,  0.0, 0.0, 1.0]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [0, 1, 2]

    return Shape(vertices, indices)


def createRainbowQuad():

    # Defining the location and colors of each vertex  of the shape
    vertices = [
    #   positions        colors
        -0.5, -0.5, 0.0,  1.0, 0.0, 0.0,
         0.5, -0.5, 0.0,  0.0, 1.0, 0.0,
         0.5,  0.5, 0.0,  0.0, 0.0, 1.0,
        -0.5,  0.5, 0.0,  1.0, 1.0, 1.0]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
        0, 1, 2,
        2, 3, 0]

    return Shape(vertices, indices)


def createColorQuad(r, g, b):

    # Defining locations and colors for each vertex of the shape    
    vertices = [
    #   positions        colors
        -0.5, -0.5, 0.0,  r, g, b,
         0.5, -0.5, 0.0,  r, g, b,
         0.5,  0.5, 0.0,  r, g, b,
        -0.5,  0.5, 0.0,  r, g, b]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
         0, 1, 2,
         2, 3, 0]

    return Shape(vertices, indices)

def createCilindro(r, g, b):

# Defining the location and colors of each vertex of the shape
    vertices = [ 0.0, 0.5, 0.0, r, g, b, 0.0, -0.5, 0.0, r, g, b ]
    indices = []
    n = 180
    for i in range(n):
        theta = (2*np.pi/n)*i
        theta_next = (2*np.pi/n)*(i+1)
        vertices += [0.5*np.cos(theta), 0.5, 0.5*np.sin(theta), r, g, b] 
        vertices += [0.5*np.cos(theta), -0.5, 0.5*np.sin(theta), r, g, b]
        vertices += [0.5*np.cos(theta_next), 0.5, 0.5*np.sin(theta_next), r, g, b] 
        vertices += [0.5*np.cos(theta_next), -0.5, 0.5*np.sin(theta_next), r, g, b]
        indices += [ 0, 4*i+2, 4*i+4, 1, 4*i+3, 4*i+5, 4*i+2,4*i+3,4*i+4, 4*i+4, 4*i+5, 4*i+3 ]

    return Shape(vertices, indices)


def createTextureQuad(image_filename, nx=1, ny=1):

    # Defining locations and texture coordinates for each vertex of the shape    
    vertices = [
    #   positions        texture
        -0.5, -0.5, 0.0,  0, 0,
         0.5, -0.5, 0.0, nx, 0,
         0.5,  0.5, 0.0, nx, ny,
        -0.5,  0.5, 0.0,  0, ny]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
         0, 1, 2,
         2, 3, 0]

    textureFileName = image_filename

    return Shape(vertices, indices, textureFileName)


def createRainbowCube():

    # Defining the location and colors of each vertex  of the shape
    vertices = [
    #    positions         colors
        -0.5, -0.5,  0.5,  1.0, 0.0, 0.0,
         0.5, -0.5,  0.5,  0.0, 1.0, 0.0,
         0.5,  0.5,  0.5,  0.0, 0.0, 1.0,
        -0.5,  0.5,  0.5,  1.0, 1.0, 1.0,
 
        -0.5, -0.5, -0.5,  1.0, 1.0, 0.0,
         0.5, -0.5, -0.5,  0.0, 1.0, 1.0,
         0.5,  0.5, -0.5,  1.0, 0.0, 1.0,
        -0.5,  0.5, -0.5,  1.0, 1.0, 1.0]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
         0, 1, 2, 2, 3, 0,
         4, 5, 6, 6, 7, 4,
         4, 5, 1, 1, 0, 4,
         6, 7, 3, 3, 2, 6,
         5, 6, 2, 2, 1, 5,
         7, 4, 0, 0, 3, 7]

    return Shape(vertices, indices)


def createColorCube(r, g, b):

    # Defining the location and colors of each vertex  of the shape
    vertices = [
    #    positions        colors
        -0.5, -0.5,  0.5, r, g, b,
         0.5, -0.5,  0.5, r, g, b,
         0.5,  0.5,  0.5, r, g, b,
        -0.5,  0.5,  0.5, r, g, b,

        -0.5, -0.5, -0.5, r, g, b,
         0.5, -0.5, -0.5, r, g, b,
         0.5,  0.5, -0.5, r, g, b,
        -0.5,  0.5, -0.5, r, g, b]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
         0, 1, 2, 2, 3, 0,
         4, 5, 6, 6, 7, 4,
         4, 5, 1, 1, 0, 4,
         6, 7, 3, 3, 2, 6,
         5, 6, 2, 2, 1, 5,
         7, 4, 0, 0, 3, 7]

    return Shape(vertices, indices)

def createColorTriangularPrism(r, g, b):

    # Defining the location and colors of each vertex  of the shape
    vertices = [
    #    positions        colors
         0.5, -0.5, -0.5, r, g, b,
         0.5,  0.5, -0.5, r, g, b,
        -0.5,  0.5,  0.5, r, g, b,
        -0.5, -0.5,  0.5, r, g, b,

        -0.5,  0.5, -0.5, r, g, b,
        -0.5, -0.5, -0.5, r, g, b,]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
         0, 1, 2, 2, 3, 0,
         1, 4, 2,
         1, 4, 5, 4, 5, 0,
         0, 5, 3,
         4, 2, 3, 2, 3, 5]

    return Shape(vertices, indices)


def createTextureCube(image_filename):

    # Defining locations and texture coordinates for each vertex of the shape  
    vertices = [
    #   positions         texture coordinates
    # Z+
        -0.5, -0.5,  0.5, 0, 0,
         0.5, -0.5,  0.5, 1, 0,
         0.5,  0.5,  0.5, 1, 1,
        -0.5,  0.5,  0.5, 0, 1,

    # Z-
        -0.5, -0.5, -0.5, 0, 0,
         0.5, -0.5, -0.5, 1, 0,
         0.5,  0.5, -0.5, 1, 1,
        -0.5,  0.5, -0.5, 0, 1,
        
    # X+
         0.5, -0.5, -0.5, 0, 0,
         0.5,  0.5, -0.5, 1, 0,
         0.5,  0.5,  0.5, 1, 1,
         0.5, -0.5,  0.5, 0, 1
,
 
    # X-
        -0.5, -0.5, -0.5, 0, 0,
        -0.5,  0.5, -0.5, 1, 0,
        -0.5,  0.5,  0.5, 1, 1,
        -0.5, -0.5,  0.5, 0, 1,

    # Y+
        -0.5,  0.5, -0.5, 0, 0,
         0.5,  0.5, -0.5, 1, 0,
         0.5,  0.5,  0.5, 1, 1,
        -0.5,  0.5,  0.5, 0, 1,

    # Y-
        -0.5, -0.5, -0.5, 0, 0,
         0.5, -0.5, -0.5, 1, 0,
         0.5, -0.5,  0.5, 1, 1,
        -0.5, -0.5,  0.5, 0, 1
        ]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
          0, 1, 2, 2, 3, 0, # Z+
          7, 6, 5, 5, 4, 7, # Z-
          8, 9,10,10,11, 8, # X+
         15,14,13,13,12,15, # X-
         19,18,17,17,16,19, # Y+
         20,21,22,22,23,20] # Y-

    return Shape(vertices, indices, image_filename)


def createRainbowNormalsCube():

    sq3 = 0.57735027

    # Defining the location and colors of each vertex  of the shape
    vertices = [
            -0.5, -0.5,  0.5, 1.0, 0.0, 0.0, -sq3, -sq3, sq3,
             0.5, -0.5,  0.5, 0.0, 1.0, 0.0,  sq3, -sq3,  sq3,
             0.5,  0.5,  0.5, 0.0, 0.0, 1.0,  sq3,  sq3,  sq3,
            -0.5,  0.5,  0.5, 1.0, 1.0, 1.0, -sq3,  sq3,  sq3,

            -0.5, -0.5, -0.5, 1.0, 1.0, 0.0, -sq3, -sq3, -sq3,
             0.5, -0.5, -0.5, 0.0, 1.0, 1.0,  sq3, -sq3, -sq3,
             0.5,  0.5, -0.5, 1.0, 0.0, 1.0,  sq3,  sq3, -sq3,
            -0.5,  0.5, -0.5, 1.0, 1.0, 1.0, -sq3,  sq3, -sq3]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [0, 1, 2, 2, 3, 0,
               4, 5, 6, 6, 7, 4,
               4, 5, 1, 1, 0, 4,
               6, 7, 3, 3, 2, 6,
               5, 6, 2, 2, 1, 5,
               7, 4, 0, 0, 3, 7]

    return Shape(vertices, indices)

def createColorNormalTriangularPrism(r, g, b):

    r2 = 1 / np.sqrt(2)
    r10 = 10 / np.sqrt(10)
    # Defining the location and colors of each vertex  of the shape
    vertices = [
    #    positions        colors    normals
    # XZ+
         0.5, -0.5, -0.5, r, g, b, r2, 0, r2, # 0
         0.5,  0.5, -0.5, r, g, b, r2, 0, r2, # 1
        -0.5,  0.5,  0.5, r, g, b, r2, 0, r2, # 2
        -0.5, -0.5,  0.5, r, g, b, r2, 0, r2, # 3

    # Y+
        0.5, 0.5, -0.5, r, g, b, 0, 1, 0, # 4
        -0.5, 0.5, -0.5, r, g, b, 0, 1, 0,# 5
        -0.5, 0.5, 0.5, r, g, b, 0, 1, 0,# 6

    # Y-
        0.5, -0.5, -0.5, r, g, b, 0, -1, 0, # 7
        -0.5, -0.5, -0.5, r, g, b, 0, -1, 0, # 8
        -0.5, -0.5, 0.5, r, g, b, 0, -1, 0, # 9

    # X-
        -0.5, 0.5, -0.5, r, g, b, -1, 0, 0, # 10
        -0.5, 0.5, 0.5, r, g, b, -1, 0, 0, # 11
        -0.5, -0.5, 0.5, r, g, b, -1, 0, 0, # 12
        -0.5, -0.5, -0.5, r, g, b, -1, 0, 0, # 13

    # Z-
        0.5, 0.5, -0.5, r, g, b, 0, 0, -1, # 14
        -0.5, 0.5, -0.5, r, g, b, 0, 0, -1, # 15
        -0.5, -0.5, -0.5, r, g, b, 0, 0, -1, # 16
        0.5, -0.5, -0.5, r, g, b, 0, 0, -1, # 17
        ]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
         0, 1, 2, 2, 3, 0,
         4, 5, 6,
         7, 8, 9,
         10, 11, 12, 12, 13, 10,
         14, 15, 16, 16, 17, 14]

    return Shape(vertices, indices)

def createColorNormalsCube(r, g, b):

    # Defining the location and colors of each vertex  of the shape
    vertices = [
    #   positions         colors   normals
    # Z+
        -0.5, -0.5,  0.5, r, g, b, 0,0,1,
         0.5, -0.5,  0.5, r, g, b, 0,0,1,
         0.5,  0.5,  0.5, r, g, b, 0,0,1,
        -0.5,  0.5,  0.5, r, g, b, 0,0,1,

    # Z-
        -0.5, -0.5, -0.5, r, g, b, 0,0,-1,
         0.5, -0.5, -0.5, r, g, b, 0,0,-1,
         0.5,  0.5, -0.5, r, g, b, 0,0,-1,
        -0.5,  0.5, -0.5, r, g, b, 0,0,-1,
        
    # X+
        0.5, -0.5, -0.5, r, g, b, 1,0,0,
        0.5,  0.5, -0.5, r, g, b, 1,0,0,
        0.5,  0.5,  0.5, r, g, b, 1,0,0,
        0.5, -0.5,  0.5, r, g, b, 1,0,0,
 
    # X-
        -0.5, -0.5, -0.5, r, g, b, -1,0,0,
        -0.5,  0.5, -0.5, r, g, b, -1,0,0,
        -0.5,  0.5,  0.5, r, g, b, -1,0,0,
        -0.5, -0.5,  0.5, r, g, b, -1,0,0,

    # Y+
        -0.5, 0.5, -0.5, r, g, b, 0,1,0,
         0.5, 0.5, -0.5, r, g, b, 0,1,0,
         0.5, 0.5,  0.5, r, g, b, 0,1,0,
        -0.5, 0.5,  0.5, r, g, b, 0,1,0,

    # Y-
        -0.5, -0.5, -0.5, r, g, b, 0,-1,0,
         0.5, -0.5, -0.5, r, g, b, 0,-1,0,
         0.5, -0.5,  0.5, r, g, b, 0,-1,0,
        -0.5, -0.5,  0.5, r, g, b, 0,-1,0
        ]

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
          0, 1, 2, 2, 3, 0, # Z+
          7, 6, 5, 5, 4, 7, # Z-
          8, 9,10,10,11, 8, # X+
         15,14,13,13,12,15, # X-
         19,18,17,17,16,19, # Y+
         20,21,22,22,23,20] # Y-

    return Shape(vertices, indices)


def createTextureNormalsCube(image_filename):

    # Defining locations,texture coordinates and normals for each vertex of the shape  
    vertices = [
    #   positions            tex coords   normals
    # Z+
        -0.5, -0.5,  0.5,    0, 0,        0,0,1,
         0.5, -0.5,  0.5,    1, 0,        0,0,1,
         0.5,  0.5,  0.5,    1, 1,        0,0,1,
        -0.5,  0.5,  0.5,    0, 1,        0,0,1,   
    # Z-          
        -0.5, -0.5, -0.5,    0, 0,        0,0,-1,
         0.5, -0.5, -0.5,    1, 0,        0,0,-1,
         0.5,  0.5, -0.5,    1, 1,        0,0,-1,
        -0.5,  0.5, -0.5,    0, 1,        0,0,-1,
       
    # X+          
         0.5, -0.5, -0.5,    0, 0,        1,0,0,
         0.5,  0.5, -0.5,    1, 0,        1,0,0,
         0.5,  0.5,  0.5,    1, 1,        1,0,0,
         0.5, -0.5,  0.5,    0, 1,        1,0,0,   
    # X-          
        -0.5, -0.5, -0.5,    0, 0,        -1,0,0,
        -0.5,  0.5, -0.5,    1, 0,        -1,0,0,
        -0.5,  0.5,  0.5,    1, 1,        -1,0,0,
        -0.5, -0.5,  0.5,    0, 1,        -1,0,0,   
    # Y+          
        -0.5,  0.5, -0.5,    0, 0,        0,1,0,
         0.5,  0.5, -0.5,    1, 0,        0,1,0,
         0.5,  0.5,  0.5,    1, 1,        0,1,0,
        -0.5,  0.5,  0.5,    0, 1,        0,1,0,   
    # Y-          
        -0.5, -0.5, -0.5,    0, 0,        0,-1,0,
         0.5, -0.5, -0.5,    1, 0,        0,-1,0,
         0.5, -0.5,  0.5,    1, 1,        0,-1,0,
        -0.5, -0.5,  0.5,    0, 1,        0,-1,0
        ]   

    # Defining connections among vertices
    # We have a triangle every 3 indices specified
    indices = [
          0, 1, 2, 2, 3, 0, # Z+
          7, 6, 5, 5, 4, 7, # Z-
          8, 9,10,10,11, 8, # X+
         15,14,13,13,12,15, # X-
         19,18,17,17,16,19, # Y+
         20,21,22,22,23,20] # Y-

    return Shape(vertices, indices, image_filename)