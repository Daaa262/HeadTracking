import cv2
import mediapipe as mp

# Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5)

# Image
image = cv2.imread("test4.jpg")
h, w, _ = image.shape
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Facial landmarks
result = face_mesh.process(rgb_image)

for facial_landmarks in result.multi_face_landmarks:
    for i in range(450):
        x = facial_landmarks.landmark[i].x * w
        y = facial_landmarks.landmark[i].y * h
        if i == 164 or i == 8:
            cv2.circle(image, (x.__int__(), y.__int__()), 2, (0, 255, 0), -1)
        else:
            cv2.circle(image, (x.__int__(), y.__int__()), 2, (0, 0, 0), -1)

cv2.imshow("Testowe", image)
cv2.waitKey(0)