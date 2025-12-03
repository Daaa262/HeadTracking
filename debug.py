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
    snapshot = shared_viewpoint[:]

    def offaxis_frustum_matrix(l, r, b, t, n, f):
        """Standardowa macierz proj. off-axis (row-major numpy)"""
        m = numpy.zeros((4, 4), dtype=numpy.float32)
        m[0, 0] = 2.0 * n / (r - l)
        m[0, 2] = (r + l) / (r - l)
        m[1, 1] = 2.0 * n / (t - b)
        m[1, 2] = (t + b) / (t - b)
        m[2, 2] = -(f + n) / (f - n)
        m[2, 3] = -2.0 * f * n / (f - n)
        m[3, 2] = -1.0
        return m

    def view_matrix_screen_aligned(eye):
        """
        Buduje macierz widoku (world->eye space) zakładając, że ekran jest w z=0
        i jego osie to globalne X (poziom) i Y (pion). Zdefiniujemy bazę ekranu:
          ex = (1,0,0), ey = (0,1,0), n = (0,0,-1)  (n skierowana od ekranu do oka)
        Zwraca 4x4 macierz row-major (numpy), gotową do transpozycji przed glLoadMatrixf.
        """
        ex = numpy.array([1.0, 0.0, 0.0], dtype=numpy.float32)
        ey = numpy.array([0.0, 1.0, 0.0], dtype=numpy.float32)
        n = numpy.array([0.0, 0.0, 1.0], dtype=numpy.float32)  # normalny od ekranu w stronę oka

        R = numpy.stack([ex, ey, n], axis=1)  # 3x3, kolumny = wektory bazowe
        Rt = R.T  # inverse rotation = R^T (bo R ortonormalna)

        T = - Rt @ eye  # translacja: -R^T * eye
        M = numpy.eye(4, dtype=numpy.float32)
        M[:3, :3] = Rt.astype(numpy.float32)
        M[:3, 3] = T.astype(numpy.float32)
        return M

    def make_view_and_projection(w, h, eye, near=10.0, far=10000.0):
        """
        w, h      : rozmiary monitora w mm
        eye       : numpy array [Cx, Cy, Cz] (mm), Cz MUST be > 0
        near, far : near/far planes (mm)
        Zwraca (proj_matrix, view_matrix) w formacie numpy row-major (4x4).
        Przy ładowaniu do OpenGL z PyOpenGL użyj glLoadMatrixf(mat.T).
        """
        Cx, Cy, Cz = float(eye[0]), float(eye[1]), float(eye[2])

        # narożniki ekranu w z=0
        L = -w / 2.0
        R = w / 2.0
        B = -h / 2.0
        T = h / 2.0

        # oblicz l,r,b,t na near plane (prosty wzór perspektywy)
        l = (L - Cx) * (near / Cz)
        r = (R - Cx) * (near / Cz)
        b = (B - Cy) * (near / Cz)
        t = (T - Cy) * (near / Cz)

        proj = offaxis_frustum_matrix(l, r, b, t, near, far)
        view = view_matrix_screen_aligned(numpy.array(eye, dtype=numpy.float32))

        return proj, view

    proj_matrix, view_matrix = make_view_and_projection(Config.Screen.width_mm, Config.Screen.height_mm, snapshot[:3], near=10.0, far=10000.0)

    glMatrixMode(GL_PROJECTION)
    glLoadMatrixf(proj_matrix.T)   # transpose potrzebny bo OpenGL oczekuje column-major
    glMatrixMode(GL_MODELVIEW)
    glLoadMatrixf(view_matrix.T)

    # glMatrixMode(GL_PROJECTION)
    # glLoadIdentity()
    # glFrustum(snapshot[3], snapshot[4], snapshot[5], snapshot[6], snapshot[7], snapshot[8])

    # glLoadMatrixf([
    #     2*near/(right-left), 0, 0, 0,
    #     0, 2*near/(top-bottom), 0, 0,
    #     (right + left) / (right - left), (top + bottom) / (top - bottom), -(far + near) / (far - near), -1,
    #     0, 0, -2 * far * near / (far - near), 0])

    # glMatrixMode(GL_MODELVIEW)
    # glLoadIdentity()
    # gluLookAt(
    #     snapshot[0], snapshot[1], snapshot[2],
    #     0.0, 0.0, 0.0,
    #     0.0, 1.0, 0.0
    # )

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

    draw_text(10, Config.Screen.height - 20,  f"Position:")
    draw_text(10, Config.Screen.height - 40,  f"  x:                {shared_viewpoint[0]:8.3f}")
    draw_text(10, Config.Screen.height - 60,  f"  y:                {shared_viewpoint[1]:8.3f}")
    draw_text(10, Config.Screen.height - 80,  f"  z:                {shared_viewpoint[2]:8.3f}")

    draw_text(10, Config.Screen.height - 100, f"Frustum:")
    draw_text(10, Config.Screen.height - 120, f"  left:             {shared_viewpoint[3]:8.3f}")
    draw_text(10, Config.Screen.height - 140, f"  right:            {shared_viewpoint[4]:8.3f}")
    draw_text(10, Config.Screen.height - 160, f"  bottom:           {shared_viewpoint[5]:8.3f}")
    draw_text(10, Config.Screen.height - 180, f"  top:              {shared_viewpoint[6]:8.3f}")
    draw_text(10, Config.Screen.height - 200, f"  near:             {shared_viewpoint[7]:8.3f}")
    draw_text(10, Config.Screen.height - 220, f"  far:              {shared_viewpoint[8]:8.3f}")

    draw_text(10, Config.Screen.height - 240, f"Dynamic Data:")
    draw_text(10, Config.Screen.height - 260, f"  smoothing_factor:{shared_dynamic_data['smoothing_factor'][0]:8.6f}")
    draw_text(10, Config.Screen.height - 280, f"  camera_fps:      {shared_dynamic_data['camera_fps'][0]}")
    draw_text(10, Config.Screen.height - 300, f"  face_mesh_fps:   {shared_dynamic_data['face_mesh_fps'][0]}")
    draw_text(10, Config.Screen.height - 320, f"  viewpoint_fps:   {shared_dynamic_data['viewpoint_fps'][0]}")

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


def draw_monitor_frame():
    glDisable(GL_LIGHTING)
    glColor3f(0.0, 1.0, 0.0)
    glLineWidth(3.0)

    w = Config.Screen.width_mm / 2.0
    h = Config.Screen.height_mm / 2.0

    for i in range(10):
        glBegin(GL_LINE_LOOP)
        glVertex3f(-w, -h, i * -100.0)
        glVertex3f(w, -h, i * -100.0)
        glVertex3f(w, h, i * -100.0)
        glVertex3f(-w, h, i * -100.0)
        glEnd()

    glEnable(GL_LIGHTING)
    glLineWidth(1.0)

def display():
    global alpha, now
    alpha = alpha + (time.time() - now) * 100
    now = time.time()

    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    update_projection()
    draw_shapes()
    draw_monitor_frame()
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