import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
import socket
import json

from IPython.display import display, Image, clear_output
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
clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect(('10.10.10.10',8089))
#message = 'hello'
#clientsocket.send(message.encode('utf-8'))

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (255, 255, 255)  # white

#cv2.namedWindow("Control")

def draw_landmarks_on_image(rgb_image, detection_result):
    #gesture_text = detection_result.gestures[0][0].category_name if detection_result.gestures else "No Gesture"
    # Draw the hand annotations on the image.
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    #print(gesture_text)
    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
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
        #text_x = int(min(x_coordinates) * width)
        #text_y = int(min(y_coordinates) * height) - MARGIN
        # Draw handedness (left or right hand) and detected gesture on the image.
        #cv2.putText(annotated_image, f"{handedness[0].category_name} {gesture_text}",
                    # (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    # FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        print(detection_result.hand_landmarks[0][8].z)

        comands={
            "dir": "none"
            }
        
        yPos = detection_result.hand_landmarks[0][8].y
        xPos = detection_result.hand_landmarks[0][8].x
        #Right/Left
        if(0.5 > yPos > 0.2):
            if (xPos > 0.7):
                print("Left")
                comands.update({"dir": "left"})
            elif (xPos< 0.3):
                print("Right")
                comands.update({"dir":"right"})
            else: comands.update({"dir":"none"})

        #Forward/Backward
        if(0.3 < xPos < 0.7):
            if (yPos < 0.2):
                print("Forward")
                comands.update({"dir":"forward"})
            elif (yPos > 0.5):
                print("Backward")
                comands.update({"dir":"backward"})
            else: comands.update({"dir":"none"})

        
        
        message = json.dumps(comands)
        clientsocket.send(message.encode('utf-8'))
        print(message)
    return annotated_image

# Capture and display live frames from the default webcam inline in Jupyter

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
        _, buf = cv2.imencode('.jpg', annotated_image)
        
        height, width, _ = annotated_image.shape
        
        source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        destination = np.float32([[width, 0], [0, 0], [0, height], [width, height]])
        perspective_matrix = cv2.getPerspectiveTransform(source, destination)
        warped = cv2.warpPerspective(annotated_image, perspective_matrix, (width, height))
        
        cv2.imshow("Gesture Image", warped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # display(Image(data=buf.tobytes()))
        # clear_output(wait=True)

        

        time.sleep(0.03)  # ~30 FPS-ish; adjust as needed
except KeyboardInterrupt:
    # Stop the loop with Ctrl+C in the notebook
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    # Only for debuging:  # print(recognition_result)
    print("Webcam stopped and released.")
    clientsocket.send("close".encode('utf-8'))