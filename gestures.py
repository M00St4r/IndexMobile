import mediapipe as mp
import cv2
import numpy as np

import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
mp.__version__

model_path = '../IndexMobile/models/gesture_recognizer.task'

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]
    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())
    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN
    # Draw handedness (left or right hand) and detected gesture on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name} {gesture_text}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    print(gesture_text)
    return annotated_image

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam (VideoCapture(0) failed)")

start = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        timestamp_ms = int((time.time() - start) * 1000)
        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        recognition_result = recognizer.recognize(mp_image)
        annotated_image = draw_landmarks_on_image(frame, recognition_result)
        # Encode as JPEG and display inline (fast and avoids creating new matplotlib figures)
        #_, buf = cv2.imencode('.jpg', annotated_image)
        time.sleep(0.03)  # ~30 FPS-ish; adjust as needed
except KeyboardInterrupt:
    # Stop the loop with Ctrl+C in the notebook
    pass
finally:
    cap.release()
    # Only for debuging:  # print(recognition_result)
    print("Webcam stopped and released.")