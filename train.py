# TRAIN CLASSIFICATION MODEL
import os
import numpy as np
import joblib

from tensorflow.keras.models import load_model

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from helpers import generate_data, progress

# ---lOAD MODELS---
face_model = load_model('./models/facenet_keras.h5')

# ---GENERATING DATA---
data_folder = './data/'

# Paths to training images
image_paths = [data_folder + file for file in os.listdir(data_folder)]

print("Generating Dataset...")

# Generate data from paths
dataset = [generate_data(face_model, path) for path in image_paths]

X = np.array([x[0] for x in dataset])
y = np.array([x[1] for x in dataset])

# ---TRAINING---
print("Dataset generated. Start training...")

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# SVM Classifer
classify_model = SVC(kernel='linear', probability=True)
classify_model.fit(X, y)

# Mapper from encoded labels to label names
mapper = {i: name for i, name in enumerate(encoder.classes_)}

# Save model and mapper
joblib.dump(classify_model, 'models/classify_model.pkl')
joblib.dump(mapper, 'models/mapper.pkl')

print("Training completed! Model saved!")
