import cv2
import numpy as np
import dlib

def extractFirstFrame(videoDir):
    cap = cv2.VideoCapture(videoDir)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return None
    
    cap.release()
    cv2.destroyAllWindows()

    return frame


def detect_face(videoDir):
    first_frame = extractFirstFrame(videoDir)
    detector = dlib.get_frontal_face_detector()
    gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray_frame, 1)

    if len(faces) > 0:
        face = faces[0]

        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        center_x, center_y = x+w/2, y+h/2

        diagonal_length = np.sqrt(w**2+h**2)

        scaling_factor = 1/diagonal_length

        return (center_x, center_y), scaling_factor, videoDir
    else:
        print("No face detected.")
        return None, None
