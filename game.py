import json
import math
import socket
import time
from random import random, sample

import numpy
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.raw.GLU import gluPerspective
from screeninfo import get_monitors

MONITOR_SIZE = [get_monitors()[0].width_mm, get_monitors()[0].height_mm]
GRID_X = 50
GRID_Y = 30

now = time.time()
start = None

HOST = "0.0.0.0"
PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))
sock.setblocking(False)

def get_data():
    latest_data = None
    while True:
        try:
            data, addr = sock.recvfrom(4096)
            latest_data = json.loads(data.decode("utf-8"))
        except BlockingIOError:
            break
    return latest_data

target = []
points1 = []
points2 = []
def generate_level(n):
    global target, points1, points2

    target_z = random() * 200 + 350
    target = [(random() - 0.5) * target_z / 2, (random() - 0.5) * target_z / 8 + 40, target_z]

    direction = []
    for i in range(GRID_X):
        x_pos = -MONITOR_SIZE[0] / 2 + MONITOR_SIZE[0] * i / (GRID_X - 1)
        direction.append([])
        for j in range(GRID_Y):
            y_pos = -MONITOR_SIZE[1] / 2 + MONITOR_SIZE[1] * j / (GRID_Y - 1)
            direction[i].append([x_pos - target[0], y_pos - target[1], -target[2]])
            direction[i][j] /= numpy.linalg.norm(direction[i][j])

    all_pairs = [(x, y)
                 for x in range(8, GRID_X - 8)
                 for y in range(5, GRID_Y - 5)]

    positions = sample(all_pairs, n)

    points1 = []
    points2 = []
    for i in range(n):
        distance1 = 200 + random() * 300
        distance2 = 800 + random() * 500
        points1.append(target + direction[positions[i][0]][positions[i][1]] * distance1)
        points2.append(target + direction[positions[i][0]][positions[i][1]] * distance2)

def update_projection(view, projection):
    glMatrixMode(GL_PROJECTION)
    glLoadMatrixf(projection)
    glMatrixMode(GL_MODELVIEW)
    glLoadMatrixf(view)

def draw_text(x, y, text):
    glMatrixMode(GL_MODELVIEW)
    glRasterPos2f(x, y)
    for ch in text:
        glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(ch))

def sphere_fully_blocks(blue_center, blue_radius, red_center, red_radius, viewpoint):
    vp = numpy.array([viewpoint['x'], viewpoint['y'], viewpoint['z']], dtype=float)
    b = numpy.array(blue_center, dtype=float)
    r = numpy.array(red_center, dtype=float)

    db = numpy.linalg.norm(b - vp)
    dr = numpy.linalg.norm(r - vp)
    if db <= 1e-8 or dr <= 1e-8:
        return False

    dir_b = (b - vp) / db
    dir_r = (r - vp) / dr

    cos_theta = numpy.dot(dir_b, dir_r)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.acos(cos_theta)

    def ang_radius(R, d):
        if R >= d:
            return math.pi / 2.0
        return math.asin(R / d)

    alpha_b = ang_radius(blue_radius, db)
    alpha_r = ang_radius(red_radius, dr)

    if alpha_b + 1e-12 < (theta + alpha_r):
        return False

    front_blue = db - blue_radius
    front_red = dr - red_radius

    if front_blue >= front_red - 1e-6:
        return False

    return True

correct = True
def draw_shapes(viewpoint):
    global correct

    glPushMatrix()

    if start is not None:
        front_radius = 6.0
        back_radius = 6.0

        occludes = [False] * len(points1)
        for i, blue in enumerate(points1):
            for red in points2:
                if sphere_fully_blocks(blue, front_radius, red, back_radius, viewpoint):
                    occludes[i] = True
                    break

        correct_amount = 0
        for i, point in enumerate(points1):
            if occludes[i]:
                correct_amount += 1
                glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.0, 1.0, 0.0, 1.0])
            else:
                glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [1.0, 0.0, 0.0, 1.0])
            glPushMatrix()
            glTranslatef(float(point[0]), float(point[1]), float(point[2]))
            glutSolidSphere(front_radius, 12, 12)
            glPopMatrix()

        if correct_amount == len(points1):
            correct = True
        else:
            correct = False

        glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [0.0, 0.0, 1.0, 1.0])
        for point in points2:
            glPushMatrix()
            glTranslatef(point[0], point[1], point[2])
            glutSolidSphere(back_radius, 12, 12)
            glPopMatrix()

    glPopMatrix()

current_viewpoint = {"x": 0, "y": 0, "z": 0}
def display():
    global current_viewpoint

    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    latest_data = get_data()
    if latest_data is not None:
        current_viewpoint = latest_data.get("viewpoint")
        update_projection(latest_data.get("view_matrix"), latest_data.get("projection_matrix"))

    draw_shapes(current_viewpoint)

    glutSwapBuffers()

amount = 0
def keyboard(key, x, y):
    global start, amount, correct

    if key == b's' and correct:
        correct = False
        if amount == 0 or amount == 32:
            amount = 1
        else:
            amount *= 2
        generate_level(amount)
        start = time.time()
    elif key == b'q':
        glutLeaveMainLoop()

def init():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB | GLUT_MULTISAMPLE)
    glutInitWindowSize(1920, 1080)
    glutCreateWindow(b"Game")

    glEnable(GL_NORMALIZE)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_POLYGON_SMOOTH)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glFrontFace(GL_CCW)

    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.0, 0.0, 0.0, 0.6])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.6, 0.6, 0.6, 0.6])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 0.6])

    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.3, 0.0, 0.2, 0.6])
    glMaterialfv(GL_FRONT, GL_SHININESS, 50.0)

    glutIdleFunc(display)
    glutDisplayFunc(display)
    glutKeyboardFunc(keyboard)
    glutFullScreen()

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, 1920 / 1080, 0.1, 1000.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glutMainLoop()

if __name__ == "__main__":
    init()