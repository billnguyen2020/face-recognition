# DETECT AND CLASSIFY FACES
import joblib
import os
import cv2
import dlib
import numpy as np
from imutils import face_utils
import time

from tensorflow import convert_to_tensor
from tensorflow.keras.models import load_model

from helpers import align_face, get_embedding

# Load Face Detection model
face_model = load_model('models/facenet_keras.h5')
net = cv2.dnn.readNet("weights/yolov3-face.weights", "cfg/yolov3-face.cfg")

# Get Output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load Classifier Model
mapper = joblib.load('models/mapper.pkl')
classify_model = joblib.load('models/classify_model.pkl')

# Face Aligner Object
shape_predictor = dlib.shape_predictor(
    'models/facial_landmarks/shape_predictor_5_face_landmarks.dat')
fa = face_utils.facealigner.FaceAligner(
    shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))

font = cv2.FONT_HERSHEY_PLAIN

# Open webcam
cap = cv2.VideoCapture(0)

starting_time = time.time()
frame_id = 0

# Start detecting...
while True:

    # Read a video frame
    _, frame = cap.read()
    frame_id += 1

    height, width, channel = frame.shape
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

    # Detect face(s)
    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []

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
        if i in indexes:  # Remove overlapped boxes
            x, y, w, h = boxes[i]

            # Align face to center of image
            fa_frame = align_face(fa, frame, x, y, w, h)

            # Reformat colors and convert to Tensor
            fa_frame = cv2.cvtColor(fa_frame, cv2.COLOR_BGR2RGB)
            fa_frame = convert_to_tensor(fa_frame, np.float32)

            # Get image embedding
            embedding = get_embedding(face_model, fa_frame).reshape(1, -1)

            # Classify image
            prediction = classify_model.predict_proba(embedding)
            label = mapper[np.argmax(prediction)]
            prob = prediction.max()

            # Draw box
            if prob > 0.5:
                color = np.random.uniform((0, 255, 3))
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label + " " + str(round(prob, 2)),
                            (x, y + 30), font, 3, color, 3)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time

    # Show FPS
    cv2.putText(frame, "FPS: " + str(round(fps, 2)),
                (10, 50), font, 2, (0, 0, 0), 3)

    # Display image
    cv2.imshow("Image", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Esc key
        break
