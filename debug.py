from config import Config

import numpy
from multiprocessing.shared_memory import SharedMemory
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

alpha = 0.0
shared_viewpoint = None

def update_projection():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    camera_position = numpy.array(shared_viewpoint[:3], dtype=float)

    forward = - camera_position
    forward /= numpy.linalg.norm(forward)

    world_up = numpy.array([0.0, 1.0, 0.], dtype=float)
    world_up /= numpy.linalg.norm(world_up)

    right = numpy.cross(forward, world_up)
    right /= numpy.linalg.norm(right)
    up = numpy.cross(right, forward)
    up /= numpy.linalg.norm(up)

    hw, hh = Config.Screen.width_mm / 2.0, Config.Screen.height_mm / 2.0
    corners = [
        right * hw + up * hh,
        -right * hw + up * hh,
        right * hw - up * hh,
        -right * hw - up * hh
    ]

    xs, ys = [], []
    for p in corners:
        v = p - camera_position
        xs.append(numpy.dot(right, v))
        ys.append(numpy.dot(up, v))

    monitor_distance = -numpy.dot(forward, camera_position)

    near = Config.Other.frustumNear
    far = Config.Other.frustumFar
    scale = near / monitor_distance

    left = min(xs) * scale
    right = max(xs) * scale
    bottom = min(ys) * scale
    top = max(ys) * scale

    glFrustum(left, right, bottom, top, near, far)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(
        camera_position[0], camera_position[1], camera_position[2],
        0.0, 0.0, 0.0,
        up[0], up[1], up[2]
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

    # Wyłącz oświetlenie i głębię dla HUD
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)

    # Ustaw kolor tekstu
    glColor3f(1.0, 1.0, 1.0)

    # Wyświetl tekst
    draw_text(10, Config.Screen.height - 20, f"Position:")
    draw_text(10, Config.Screen.height - 40, f"  x:                {shared_viewpoint[0]:8.3f}")
    draw_text(10, Config.Screen.height - 60, f"  y:                {shared_viewpoint[1]:8.3f}")
    draw_text(10, Config.Screen.height - 80, f"  z:                {shared_viewpoint[2]:8.3f}")
    draw_text(10, Config.Screen.height - 100, f"Frustum:")
    draw_text(10, Config.Screen.height - 120, f"  left:            {shared_viewpoint[3]:8.3f}")
    draw_text(10, Config.Screen.height - 140, f"  right:           {shared_viewpoint[4]:8.3f}")
    draw_text(10, Config.Screen.height - 160, f"  bottom:          {shared_viewpoint[5]:8.3f}")
    draw_text(10, Config.Screen.height - 180, f"  top:             {shared_viewpoint[6]:8.3f}")
    draw_text(10, Config.Screen.height - 200, f"  near:            {shared_viewpoint[7]:8.3f}")
    draw_text(10, Config.Screen.height - 220, f"  far:             {shared_viewpoint[8]:8.3f}")
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
    glScalef(1000.0, 1000.0, 1000.0)
    glRotatef(alpha, 0.0, 1.0, 0.0)

    glPushMatrix()
    glTranslatef(-0.25, 0.0, 0.0)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.9, 0.3, 0.3, 1.0])
    glutSolidCube(0.12)
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0.0, 0.05, 0.0)
    glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 64.0)
    glutSolidSphere(0.08, 256, 256)
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0.25, -0.03, 0.0)
    glRotatef(-90, 1, 0, 0)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.95, 0.85, 0.3, 1.0])
    glutSolidCone(0.06, 0.15, 32, 32)
    glPopMatrix()

    glPushMatrix()
    glTranslatef(0.0, -0.05, 0.0)
    glRotatef(90, 1, 0, 0)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.4, 0.95, 0.5, 1.0])
    glutSolidTorus(0.02, 0.06, 32, 48)
    glPopMatrix()

    glPushMatrix()
    glTranslatef(-1.8, -0.12, -2.5)
    glRotatef(alpha, 1, -1, 0)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.5, 0.9, 0.9, 1.0])
    glutSolidTetrahedron()
    glPopMatrix()

    glPushMatrix()
    glTranslatef(1.8, -0.02, -1.5)
    glRotatef(alpha, 1, 1, 0)
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.5, 0.9, 0.9, 1.0])
    glutSolidTetrahedron()
    glPopMatrix()

    glPopMatrix()

def display():
    global alpha
    alpha = (alpha + 1) % 360.0

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

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_POLYGON_SMOOTH)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    light_pos = [0.5, 1.5, 1.0, 1.0]
    glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
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

    init()