import math
import time
from random import random
import numpy
from multiprocessing.shared_memory import SharedMemory
from OpenGL.GL import *
from OpenGL.GLUT import *

alpha = 0.0
shared_viewpoint = None
snapshot = None
now = time.time()
lock = None
shared_dynamic_data = None
asteroids_data = None
config = None

cam_pos = [0.0, 0.0, 0.0]
keys_down = set()

def key_down(key, x, y):
    global shared_dynamic_data, cam_pos

    if key == b'r':
        cam_pos = [0.0, 0.0, 0.0]
    elif key == b'1':
        shared_dynamic_data['smoothing_factor'][0] /= 2
    elif key == b'2':
        shared_dynamic_data['smoothing_factor'][0] *= 2
    elif key == b'q':
        shared_dynamic_data['running_flag'][0] = 0
        glutLeaveMainLoop()

    keys_down.add(key)

def key_up(key, x, y):
    if key in keys_down:
        keys_down.remove(key)

def update_projection():
    global snapshot
    with lock:
        snapshot = shared_viewpoint[:]

    glMatrixMode(GL_PROJECTION)
    glLoadMatrixf(snapshot[3:19])
    glMatrixMode(GL_MODELVIEW)
    glLoadMatrixf(snapshot[19:35])

def draw_text(x, y, text):
    glMatrixMode(GL_MODELVIEW)
    glRasterPos2f(x, y)
    for ch in text:
        glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(ch))

def draw_debug_text():
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, config.screen.width, 0, config.screen.height, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)

    glColor3f(1.0, 1.0, 1.0)

    if snapshot is not None:
        draw_text(10, config.screen.height - 20,  f"Position:")
        draw_text(10, config.screen.height - 40,  f"  x:                {snapshot[0]:8.3f}")
        draw_text(10, config.screen.height - 60,  f"  y:                {snapshot[1]:8.3f}")
        draw_text(10, config.screen.height - 80,  f"  z:                {snapshot[2]:8.3f}")

        draw_text(10, config.screen.height - 100, f"Projection Matrix:")
        draw_text(10, config.screen.height - 120, f"{snapshot[3]:8.3f} {snapshot[4]:8.3f} {snapshot[5]:8.3f} {snapshot[6]:8.3f}")
        draw_text(10, config.screen.height - 140, f"{snapshot[7]:8.3f} {snapshot[8]:8.3f} {snapshot[9]:8.3f} {snapshot[10]:8.3f}")
        draw_text(10, config.screen.height - 160, f"{snapshot[11]:8.3f} {snapshot[12]:8.3f} {snapshot[13]:8.3f} {snapshot[14]:8.3f}")
        draw_text(10, config.screen.height - 180, f"{snapshot[15]:8.3f} {snapshot[16]:8.3f} {snapshot[17]:8.3f} {snapshot[18]:8.3f}")

        draw_text(10, config.screen.height - 200, f"View Matrix:")
        draw_text(10, config.screen.height - 220, f"{snapshot[19]:8.3f} {snapshot[20]:8.3f} {snapshot[21]:8.3f} {snapshot[22]:8.3f}")
        draw_text(10, config.screen.height - 240, f"{snapshot[23]:8.3f} {snapshot[24]:8.3f} {snapshot[25]:8.3f} {snapshot[26]:8.3f}")
        draw_text(10, config.screen.height - 260, f"{snapshot[27]:8.3f} {snapshot[28]:8.3f} {snapshot[29]:8.3f} {snapshot[30]:8.3f}")
        draw_text(10, config.screen.height - 280, f"{snapshot[31]:8.3f} {snapshot[32]:8.3f} {snapshot[33]:8.3f} {snapshot[34]:8.3f}")

        draw_text(10, config.screen.height - 300, f"Dynamic Data:")
        draw_text(10, config.screen.height - 320, f"  smoothing_factor:{shared_dynamic_data['smoothing_factor'][0]:8.6f}")
        draw_text(10, config.screen.height - 340, f"  camera_fps:      {shared_dynamic_data['camera_fps'][0]}")
        draw_text(10, config.screen.height - 360, f"  face_mesh_fps:   {shared_dynamic_data['face_mesh_fps'][0]}")
        draw_text(10, config.screen.height - 380, f"  viewpoint_fps:   {shared_dynamic_data['viewpoint_fps'][0]}")

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def update_camera_position(dt):
    speed = 300

    if b'w' in keys_down:
        cam_pos[2] -= dt * speed
    if b's' in keys_down:
        cam_pos[2] += dt * speed
    if b'a' in keys_down:
        cam_pos[0] -= dt * speed
    if b'd' in keys_down:
        cam_pos[0] += dt * speed
    if b' ' in keys_down:
        cam_pos[1] += dt * speed
    if b'c' in keys_down:
        cam_pos[1] -= dt * speed

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

    w = config.screen.width_mm / 2.0
    h = config.screen.height_mm / 2.0

    glBegin(GL_LINE_LOOP)
    glVertex3f(-w, -h, 0.0)
    glVertex3f(w, -h, 0.0)
    glVertex3f(w, h, 0.0)
    glVertex3f(-w, h, 0.0)
    glEnd()

    glEnable(GL_LIGHTING)
    glLineWidth(1.0)

def display():
    global alpha, now

    update_camera_position(time.time() - now)

    alpha = alpha + (time.time() - now) * 100
    now = time.time()

    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    update_projection()

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glTranslatef(-cam_pos[0], -cam_pos[1], -cam_pos[2])
    draw_shapes()
    draw_monitor_frame()
    glPopMatrix()

    draw_debug_text()

    glutSwapBuffers()

def init():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB | GLUT_MULTISAMPLE)
    glutInitWindowSize(config.screen.width, config.screen.height)
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
    glutKeyboardFunc(key_down)
    glutKeyboardUpFunc(key_up)
    glutFullScreen()

    glutMainLoop()

def run(global_config, shm_dynamic_data_name, shm_viewpoint_name, lock_viewpoint):
    global lock, shared_dynamic_data, shared_viewpoint, asteroids_data, config
    config = global_config

    lock = lock_viewpoint
    shm_dynamic_data = SharedMemory(name=shm_dynamic_data_name)
    shared_dynamic_data = numpy.ndarray(
        shape=(1,),
        dtype=numpy.dtype(config.debug.dynamic_fields),
        buffer=shm_dynamic_data.buf)

    shm_viewpoint = SharedMemory(name=shm_viewpoint_name)
    shared_viewpoint = numpy.ndarray(
        shape=(35,),
        dtype=numpy.float32,
        buffer=shm_viewpoint.buf
    )

    asteroids_data = []
    for i in range(120):
        theta = 2 * math.pi * random()
        u = 2 * random() - 1
        r = math.sqrt(1 - u * u)
        asteroids_data.append([360 * random(), r * math.cos(theta), r * math.sin(theta), u, 0.75 + random() / 4, 1.5 + 2 * random()])

    init()