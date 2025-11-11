import math
import time
from random import random

from config import Config

import numpy
from multiprocessing.shared_memory import SharedMemory
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

alpha = 0.0
shared_viewpoint = None
now = time.time()

def update_projection():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glFrustum(shared_viewpoint[3], shared_viewpoint[4], shared_viewpoint[5], shared_viewpoint[6], shared_viewpoint[7], shared_viewpoint[8])

    # glLoadMatrixf([
    #     2*near/(right-left), 0, 0, 0,
    #     0, 2*near/(top-bottom), 0, 0,
    #     (right + left) / (right - left), (top + bottom) / (top - bottom), -(far + near) / (far - near), -1,
    #     0, 0, -2 * far * near / (far - near), 0])


    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(
        shared_viewpoint[0], shared_viewpoint[1], shared_viewpoint[2],
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0
    )

def draw_text(x, y, text):
    glMatrixMode(GL_MODELVIEW)
    glRasterPos2f(x, y)
    for ch in text:
        glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(ch))

def draw_debug_text():
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, Config.Screen.width, 0, Config.Screen.height, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)

    glColor3f(1.0, 1.0, 1.0)

    draw_text(10, Config.Screen.height - 20, f"Position:")
    draw_text(10, Config.Screen.height - 40, f"  x:                {shared_viewpoint[0]:8.3f}")
    draw_text(10, Config.Screen.height - 60, f"  y:                {shared_viewpoint[1]:8.3f}")
    draw_text(10, Config.Screen.height - 80, f"  z:                {shared_viewpoint[2]:8.3f}")
    draw_text(10, Config.Screen.height - 100, f"Dynamic Data:")
    draw_text(10, Config.Screen.height - 120, f"  smoothing_factor:{shared_dynamic_data['smoothing_factor'][0]:8.6f}")
    draw_text(10, Config.Screen.height - 140, f"  camera_fps:      {shared_dynamic_data['camera_fps'][0]}")
    draw_text(10, Config.Screen.height - 160, f"  face_mesh_fps:   {shared_dynamic_data['face_mesh_fps'][0]}")
    draw_text(10, Config.Screen.height - 180, f"  viewpoint_fps:   {shared_dynamic_data['viewpoint_fps'][0]}")

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def draw_shapes():
    glPushMatrix()

    light_pos = [0.0, 0.0, 0.0, 1.0]
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
    glPushMatrix()
    glMaterialfv(GL_FRONT, GL_EMISSION, [1.0, 1.0, 0.0, 1.0])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.0, 0.0, 0.0, 1.0])
    glutSolidSphere(20, 32, 32)
    glMaterialfv(GL_FRONT, GL_EMISSION, [0.0, 0.0, 0.0, 1.0])
    glPopMatrix()

    for i, asteroid_data in enumerate(asteroids_data):
        glPushMatrix()
        glRotatef(alpha / 10 * (40/(40 + i))**1.5 + asteroid_data[0], asteroid_data[1], asteroid_data[2], asteroid_data[3])
        glTranslatef(0, 0, 40 + i * 2)
        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [asteroid_data[4], asteroid_data[4], asteroid_data[4], 1.0])
        glutSolidSphere(asteroid_data[5], 12, 12)
        glPopMatrix()

    glPopMatrix()

def display():
    global alpha, now
    alpha = alpha + (time.time() - now) * 100
    now = time.time()

    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    update_projection()
    draw_shapes()
    draw_debug_text()

    glutSwapBuffers()
    pass

def keyboard(key, x, y):
    global shared_dynamic_data
    if key == b'1':
        shared_dynamic_data['smoothing_factor'][0] /= 2
    elif key == b'2':
        shared_dynamic_data['smoothing_factor'][0] *= 2
    elif key == b'q':
        shared_dynamic_data['running_flag'][0] = 0
        glutLeaveMainLoop()

def init():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB | GLUT_MULTISAMPLE)
    glutInitWindowSize(Config.Screen.width, Config.Screen.height)
    glutCreateWindow(b"HeadTrack Debug")

    glEnable(GL_NORMALIZE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_POLYGON_SMOOTH)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glFrontFace(GL_CCW)

    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.0, 0.0, 0.0, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.6, 0.6, 0.6, 1.0])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])

    glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
    glMaterialfv(GL_FRONT, GL_SHININESS, 50.0)

    glutIdleFunc(display)
    glutDisplayFunc(display)
    glutKeyboardFunc(keyboard)
    glutFullScreen()

    glutMainLoop()

def run(shm_dynamic_data_name, shm_viewpoint_name):
    shm_dynamic_data = SharedMemory(name=shm_dynamic_data_name)
    global shared_dynamic_data
    shared_dynamic_data = numpy.ndarray(
        shape=(1,),
        dtype=numpy.dtype(Config.Debug.dynamic_fields),
        buffer=shm_dynamic_data.buf)

    shm_viewpoint = SharedMemory(name=shm_viewpoint_name)
    global shared_viewpoint
    shared_viewpoint = numpy.ndarray(
        shape=(9,),
        dtype=numpy.float32,
        buffer=shm_viewpoint.buf
    )

    global asteroids_data
    asteroids_data = []
    for i in range(120):
        theta = 2 * math.pi * random()
        u = 2 * random() - 1
        r = math.sqrt(1 - u * u)
        asteroids_data.append([360 * random(), r * math.cos(theta), r * math.sin(theta), u, 0.75 + random() / 4, 1.5 + 2 * random()])

    init()