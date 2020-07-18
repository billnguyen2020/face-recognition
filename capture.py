# COLLECT FACE DATA
import cv2
import dlib
from imutils import face_utils
import numpy as np
import os
import time

from helpers import align_face

# Load Face Detection model
face_model = load_model('models/facenet_keras.h5')
net = cv2.dnn.readNet("weights/yolov3-face.weights", "cfg/yolov3-face.cfg")

# Get Output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

starting_time = time.time()

# Load Classifier Model
net = cv2.dnn.readNet("weights/yolov3-face.weights", "cfg/yolov3-face.cfg")
font = cv2.FONT_HERSHEY_PLAIN

data_root_path = './data/'

# Input class name
class_ = input("Who are you?\n")

# Get image index
image_index = len([c.split('_')[0]
                   for c in os.listdir(data_root_path) if c == class_])

# Open webcam
cap = cv2.VideoCapture(0)

starting_time = time.time()
frame_id = 0

# Start detecting...
while True:
    _, frame = cap.read()
    frame_id += 1

    # Read a video frame
    height, width, channel = frame.shape
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

    # Detect face(s)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Find overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

    for i in range(len(boxes)):
        if i in indexes: # Remove overlapped boxes
            x, y, w, h = boxes[i]
            confidence = confidences[i]

            # Draw box
            color = np.random.uniform((0, 255, 3))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)),
                        (x, y + 30), font, 3, color, 3)

            # Save image every 2 frames
            if frame_id % 2 == 0:

                # Aligh face to center of image
                fa_frame = align_face(fa, frame, x, y, w, h)

                # Resize image
                fa_frame = cv2.resize(fa_frame, (160, 160))

                # Save image
                cv2.imwrite(f'data/{class_}_{image_index}.jpg', aligned_face)
                image_index += 1

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time

    # Show FPS
    cv2.putText(frame, "FPS: " + str(round(fps, 2)),
                (10, 50), font, 2, (0, 0, 0), 3)

    # Display Image
    cv2.imshow("Image", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Esc key
        break
