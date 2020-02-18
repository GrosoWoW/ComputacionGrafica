# coding=utf-8

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys

import transformations2 as tr2
import basic_shapes as bs
import scene_graph2 as sg
import easy_shaders as es

import ex_curves
from mpl_toolkits.mplot3d import Axes3D
from ex_quad_controlled import *
from ex_aux_4 import *


# A class to store the application control
# Add follow_car option
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.showAxis = False
        self.follow_car = False
        self.lights = False
        self.camera1 = False
        self.camera2 = False
        self.camera3 = False
        self.camera4 = False
        self.cameraMovile = False
        


# we will use the global controller as communication with the callback function
controller = Controller()


def on_key(window, key, scancode, action, mods):

    if action != glfw.PRESS:
        return
    
    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_LEFT_CONTROL:
        controller.showAxis = not controller.showAxis

    elif key == glfw.KEY_ESCAPE:
        sys.exit()

    elif key == glfw.KEY_C:
        controller.follow_car = not controller.follow_car

    elif key == glfw.KEY_1:
        controller.camera1 = not controller.camera1
    
    elif key == glfw.KEY_2:
        controller.camera2 = not controller.camera2

    elif key == glfw.KEY_3:
        controller.camera3 = not controller.camera3
    
    elif key == glfw.KEY_4:
        controller.camera4 = not controller.camera4

    elif key == glfw.KEY_F:
        controller.cameraMovile = not controller.cameraMovile

    else:
        print('Unknown key')

# Create car depending of isNormal value

def createEdificio():

    gpuBase = es.toGPUShape(bs.createTextureCube("textura.jpg"), GL_REPEAT, GL_NEAREST)
    gpuLados = es.toGPUShape(bs.createTextureCube("textura.jpg"), GL_REPEAT, GL_NEAREST)
    gpuBase2 = es.toGPUShape(bs.createTextureCube("techo.jpg"), GL_REPEAT, GL_NEAREST)
    gpuAntena = es.toGPUShape(bs.createTextureCube("metal.jpg"), GL_REPEAT, GL_NEAREST)
    gpuAntena1 = es.toGPUShape(bs.createCilindro(0,0,0))

    base = sg.SceneGraphNode("base")
    base.transform = np.matmul(tr2.scale(0.8/2,0.7/2,0.1875/2), tr2.translate(0,0,0.5))
    base.childs += [gpuBase]

    lados = sg.SceneGraphNode("lados")
    lados.transform = np.matmul(tr2.translate(0.15,0,0.2), tr2.scale(0.0875/2,0.468/2,0.625/2))
    lados.childs += [gpuLados]

    lados1 = sg.SceneGraphNode("lados1") 
    lados1.transform = np.matmul(tr2.translate(-0.15,0,0.2), tr2.scale(0.0875/2,0.468/2,0.625/2))
    lados1.childs += [gpuLados]

    base2 = sg.SceneGraphNode("base2")
    base2.transform = np.matmul(tr2.scale(0.495/2,0.545/2,0.645/2), tr2.translate(0,0,0.7))
    base2.childs += [gpuBase]

    base3 = sg.SceneGraphNode("base3")
    base3.transform = np.matmul(tr2.scale(0.32/2,0.32/2,2/2), tr2.translate(0,0,0.8))
    base3.childs += [gpuBase]

    base4 = sg.SceneGraphNode("base4")
    base4.transform = np.matmul(tr2.scale(0.25/2,0.25/2,0.05/2), tr2.translate(0,0,52.5))
    base4.childs += [gpuBase2]

    base5 = sg.SceneGraphNode("base4")
    base5.transform = np.matmul(tr2.scale(0.2/2,0.2/2,0.05/2), tr2.translate(0,0,53.5))
    base5.childs += [gpuBase2]

    base6 = sg.SceneGraphNode("base4")
    base6.transform = np.matmul(tr2.scale(0.15/2,0.15/2,0.05/2), tr2.translate(0,0,54.5))
    base6.childs += [gpuBase2]

    antena = sg.SceneGraphNode("antena")
    antena.transform = np.matmul(tr2.scale(0.015,0.015,0.3), tr2.translate(0,0,4.5))
    antena.childs += [gpuAntena]

    antena1 = sg.SceneGraphNode("antena1")
    antena1.transform = np.matmul(tr2.scale(0.01,0.01,0.3), tr2.translate(0,0,5))
    antena1.childs += [gpuAntena]

    antena2 = sg.SceneGraphNode("antena2")
    antena2.transform = np.matmul(tr2.scale(0.005,0.005,0.3), tr2.translate(0,0,6))
    antena2.childs += [gpuAntena]

    lados_1 =sg.SceneGraphNode("lado_1")
    lados_1.transform = np.matmul(tr2.scale(0.1/2,0.5/2,1.2/2), tr2.translate(1.55,0,1.15))
    lados_1.childs +=[gpuLados]

    lados_2 =sg.SceneGraphNode("lado_2")
    lados_2.transform = np.matmul(tr2.scale(0.1/2,0.5/2,1.2/2), tr2.translate(-1.55,0,1.15))
    lados_2.childs +=[gpuLados]

    lados_3 =sg.SceneGraphNode("lado_2")
    lados_3.transform = np.matmul(tr2.scale(0.09/2,0.4/2,0.4/2), tr2.translate(1.65,0,5.4))
    lados_3.childs +=[gpuLados]

    lados_4 =sg.SceneGraphNode("lado_2")
    lados_4.transform = np.matmul(tr2.scale(0.09/2,0.4/2,0.4/2), tr2.translate(-1.65,0,5.4))
    lados_4.childs +=[gpuLados]
    

    edificio = sg.SceneGraphNode("edificio")
    edificio.childs += [lados]
    edificio.childs += [lados1]
    edificio.childs += [base]
    edificio.childs += [base2]
    edificio.childs += [base3]
    edificio.childs += [base4]
    edificio.childs += [base5]
    edificio.childs += [base6]
    edificio.childs += [lados_1]
    edificio.childs += [lados_2]
    edificio.childs += [lados_3]
    edificio.childs += [lados_4]
    edificio.childs += [antena]
    edificio.childs += [antena1]
    edificio.childs += [antena2]

    return edificio

def createCar(r1,g1,b1, r2, g2, b2, isNormal):
   

    gpuBlackQuad = es.toGPUShape(bs.createCilindro(0,0,0), GL_REPEAT, GL_NEAREST)
    gpuChasisQuad_color1 = es.toGPUShape(bs.createTextureCube("militar.jpg"),  GL_REPEAT, GL_NEAREST)
    gpuChasisQuad_color2 = es.toGPUShape(bs.createTextureCube("auto2.png"),  GL_REPEAT, GL_NEAREST)
    gpuChasisPrism = es.toGPUShape(bs.createColorTriangularPrism(153/255, 204/255, 255/255))
    
    # Cheating a single rueda

    auto = sg.SceneGraphNode("auto")
    auto.transform = tr2.scale(0.2,0.8,0.2)
    auto.childs += [gpuBlackQuad]

    rueda = sg.SceneGraphNode("rueda")
    rueda.transform = tr2.scale(0.2/6, 0.8/6, 0.2/6)
    rueda.childs += [gpuBlackQuad]

    ruedaRotation = sg.SceneGraphNode("ruedaRotation")
    ruedaRotation.childs += [rueda]

    # Instanciating 2 ruedas, for the front and back parts
    frontrueda = sg.SceneGraphNode("frontrueda")
    frontrueda.transform = tr2.translate(0.05,0,-0.45)
    frontrueda.childs += [ruedaRotation]

    backrueda = sg.SceneGraphNode("backrueda")
    backrueda.transform = tr2.translate(-0.05,0,-0.45)
    backrueda.childs += [ruedaRotation]
    
    # Creating the bottom chasis of the car
    bot_chasis = sg.SceneGraphNode("bot_chasis")
    bot_chasis.transform = tr2.scale(1/5,0.5/5,0.5/6)
    bot_chasis.childs += [gpuChasisQuad_color1]

    # Moving bottom chasis
    moved_b_chasis = sg.SceneGraphNode("moved_b_chasis")
    moved_b_chasis.transform = tr2.translate(0, 0, -0.4)
    moved_b_chasis.childs += [bot_chasis]

    ventana = sg.SceneGraphNode("ventana")
    ventana.transform = np.matmul(tr2.scale(0.6/4,0.3/4,0.3/4), tr2.translate(0,0,-4.5))
    ventana.childs += [gpuChasisQuad_color1]

    palito = sg.SceneGraphNode("palito")
    palito.transform = np.matmul (tr2.scale(0.2,0.01,0.01), tr2.translate(0.3,-0,-35), tr2.rotationY(np.pi/6))
    palito.childs += [gpuChasisQuad_color1]

    # Joining chasis parts
    complete_chasis = sg.SceneGraphNode("complete_chasis")
    complete_chasis.childs += [moved_b_chasis]
    complete_chasis.childs += [ventana]
    complete_chasis.childs += [palito]

    # All pieces together
    car = sg.SceneGraphNode("car")
    car.childs += [complete_chasis]
    car.childs += [frontrueda]
    car.childs += [backrueda]

    return car

# Create ground with textures
def createGround(image,trans):
    gpuGround_texture = es.toGPUShape(bs.createTextureQuad(image), GL_REPEAT, GL_NEAREST)
    ground_scaled = sg.SceneGraphNode("ground_scaled")
    ground_scaled.transform = tr2.scale(2, 2, 2)
    ground_scaled.childs += [gpuGround_texture]

    ground_rotated = sg.SceneGraphNode("ground_rotated_x")
    ground_rotated.transform = tr2.rotationX(0)
    ground_rotated.childs += [ground_scaled]

    ground = sg.SceneGraphNode("ground")
    ground.transform = tr2.translate(0, 0, trans)
    ground.childs += [ground_rotated]

    return ground

def createAmbiente(filename,x,y,z,rot):
    gpuAirport_texture = es.toGPUShape(bs.createTextureQuad(filename), GL_REPEAT, GL_LINEAR)
    ambiente_scaled = sg.SceneGraphNode("ambiente_scaled")
    ambiente_scaled.transform = tr2.scale(2, 2, 2)
    ambiente_scaled.childs += [gpuAirport_texture]

    ambiente_rotated = sg.SceneGraphNode("ambiente_rotated")
    ambiente_rotated.transform = np.matmul(tr2.rotationX(np.pi / 2), tr2.rotationY(rot))
    ambiente_rotated.childs += [ambiente_scaled]

    ambiente = sg.SceneGraphNode("ambiente")
    ambiente.transform = tr2.translate(x, y, z)
    ambiente.childs += [ambiente_rotated]

    return ambiente

if __name__ == "__main__":

    C1 = np.array([[-0.5, -0.5]]).T
    C2 = np.array([[0.5, -0.5]]).T
    C3 = np.array([[0.5, 0.5]]).T
    C4 = np.array([[1.5, 0.5]]).T
    C5 = np.array([[1.5, 1.5]]).T
    c1 = fix_data(catmull_rom(C1, C2, C3, C4, C5))


    # Initialize glfw
    if not glfw.init():
        sys.exit()

    width = 1000
    height = 1000

    window = glfw.create_window(width, height, "Empire State", None, None)

    if not window:
        glfw.terminate()
        sys.exit()

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Assembling the shader program (pipeline) with shaders (simple, texture and lights)
    mvcPipeline = es.SimpleModelViewProjectionShaderProgram()
    textureShaderProgram = es.SimpleTextureModelViewProjectionShaderProgram()
    phongPipeline = es.SimplePhongShaderProgram()




    # Setting up the clear screen color
    glClearColor(1, 1, 1, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    gpuAxis = es.toGPUShape(bs.createAxis(7))
    redCarNode = createCar(252/255,246/255,246/255, 1,0,0, controller.lights)
    blueCarNode = createCar(252/255,246/255,246/255, 0, 76/255, 153/255, False)
    groundNode = createGround("pasto.jpg",0)
    ground1Node = createGround("cielo.jpg",2)
    blueCarNode.transform = np.matmul(tr2.rotationZ(-np.pi/4), tr2.translate(5,0,5))
    baseNode = createEdificio()
    ambienteNode = createAmbiente("textura1.jpg",1,0,1,np.pi / 2)
    ambiente1Node = createAmbiente('textura1.jpg',-1,0,1,np.pi / 2)
    ambiente2Node = createAmbiente('textura1.jpg',0,1,1,0)
    ambiente3Node = createAmbiente('textura1.jpg',0,-1,1,0)

    # Define radius of the circumference
    r = 0.7

    # lookAt of normal camera
    normal_view = tr2.lookAt(
            np.array([2, 2, 3]),
            np.array([0, 0, 0]),
            np.array([0, 0, 1])
        )

    t0 = glfw.get_time()
    theta = np.pi/4
    phi = np.pi/4
    a = 1
    j = 0
    while not glfw.window_should_close(window):

        glfw.poll_events()

        # Getting the time difference from the previous iteration
        t1 = glfw.get_time()
        dt = t1 - t0
        t0 = t1

        # Telling OpenGL to use our shader program
        glUseProgram(mvcPipeline.shaderProgram)
        # Using the same view and projection matrices in the whole application
        projection = tr2.perspective(45, float(width) / float(height), 0.1, 100)
        glUniformMatrix4fv(glGetUniformLocation(mvcPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

        # Calculate coordinates of the camera and redCar
        u_px = np.cos(glfw.get_time())
        u_py = np.sin(glfw.get_time())
        x = r * u_px
        y = r * u_py

        u_tx = -u_py
        u_ty = u_px

        if controller.follow_car:
            # moving camera
            normal_view = tr2.lookAt(
                np.array([x, y, 1]),
                np.array([x + r * u_tx, y + r * u_ty, 1]),
                np.array([0, 0, 1])
            )
        elif controller.camera1:
   
            normal_view = tr2.lookAt(
            np.array([1, 1, 1]),
            np.array([0, 0, 1]),
            np.array([0, 0, 1])
            )

        elif controller.camera2:
 
            normal_view = tr2.lookAt(
            np.array([0, 1, 1]),
            np.array([0, 0, 1]),
            np.array([0, 0, 1])
            )

        elif controller.camera3:
 
            normal_view = tr2.lookAt(
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
            np.array([0, 0, 1])
            )

        elif controller.camera4:
  
            normal_view = tr2.lookAt(
            np.array([0.5, 0.5, 2]),
            np.array([0,0, 0]),
            np.array([0, 0, 1])
            )


        elif controller.cameraMovile:

            if (glfw.get_key(window, glfw.KEY_A) == glfw.PRESS):
                theta -= 2 * dt

            if (glfw.get_key(window, glfw.KEY_D) == glfw.PRESS):
                theta += 2* dt

            if (glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS and phi > 0.4):
                phi -= 2 * dt

            if (glfw.get_key(window, glfw.KEY_E) == glfw.PRESS and phi < 0.8):
                phi += 2* dt    
            
            if (glfw.get_key(window, glfw.KEY_W) == glfw.PRESS and j < 1):
                j +=0.01

            if (glfw.get_key(window, glfw.KEY_S) == glfw.PRESS and j>0):
                j -=0.01
            if (glfw.get_key(window, glfw.KEY_K) == glfw.PRESS):
               a +=0.02
            if (glfw.get_key(window, glfw.KEY_L) == glfw.PRESS ):
               a -=0.02

            eyeX = a * np.sin(theta)*np.sin(phi)
            eyeY = a * np.cos(theta)*np.sin(phi) 
            eyeZ = a * np.cos(phi) + j


            atX = 0
            atY = 0
            atZ = j

            upX = 0
            upY = 0
            upZ = a * np.sin(phi)

            eye = np.array([eyeX, eyeY, eyeZ])
            at  = np.array([atX, atY, atZ])
            up  = np.array([upX, upY, upZ])

            normal_view = tr2.lookAt(eye, at, up)

        else:
            # static camera
            normal_view = tr2.lookAt(
            np.array([1, 1, 1]),
            np.array([0, 0, 0]),
            np.array([0, 0, 1])
            )

        glUniformMatrix4fv(glGetUniformLocation(mvcPipeline.shaderProgram, "view"), 1, GL_TRUE, normal_view)

        # Using GLFW to check for input events
        glfw.poll_events()

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        if controller.showAxis:
            glUniformMatrix4fv(glGetUniformLocation(mvcPipeline.shaderProgram, "model"), 1, GL_TRUE, tr2.identity())
            mvcPipeline.drawShape(gpuAxis, GL_LINES)

        pos = int(len(c1) * t1/4 ) % len(c1)

        phi1 = np.arctan2(c1[pos][1]-c1[pos-1][1],c1[pos][0]-c1[pos-1][0])
        # Moving the red car and rotating its ruedas
        redCarNode.transform = np.matmul(tr2.translate(0, 0, 0.5), tr2.translate(c1[pos][0], c1[pos][1], 0))
        redCarNode.transform = np.matmul(redCarNode.transform, tr2.rotationZ( phi1))
        redruedaRotationNode = sg.findNode(redCarNode, "ruedaRotation")
        redruedaRotationNode.transform = tr2.rotationY(10 * glfw.get_time())

    
        # Drawing redCar using light shader
        glUseProgram(textureShaderProgram.shaderProgram)

        # Setting all uniform shader variables
        
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "view"), 1, GL_TRUE, normal_view)

        sg.drawSceneGraphNode(redCarNode, textureShaderProgram)


        # Drawing ground and ricardo using texture shader
        glUseProgram(textureShaderProgram.shaderProgram)
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "projection"), 1, GL_TRUE, projection)
        glUniformMatrix4fv(glGetUniformLocation(textureShaderProgram.shaderProgram, "view"), 1, GL_TRUE, normal_view)
        # Drawing ground
        sg.drawSceneGraphNode(groundNode, textureShaderProgram)
        sg.drawSceneGraphNode(baseNode, textureShaderProgram)
        sg.drawSceneGraphNode(ambienteNode, textureShaderProgram)
        sg.drawSceneGraphNode(ambiente1Node, textureShaderProgram)
        sg.drawSceneGraphNode(ambiente2Node, textureShaderProgram)
        sg.drawSceneGraphNode(ambiente3Node, textureShaderProgram)
        sg.drawSceneGraphNode(ground1Node, textureShaderProgram)


        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)

    glfw.terminate()