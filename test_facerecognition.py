import cv2
import os
import numpy as np

# Define the paths to the training images and labels
data_dir = 'dataset'
labels_file = 'labels.txt'

# Load the training images and labels
X_train = []
y_train = []
labels = []

for label in os.listdir(data_dir):
    label_path = os.path.join(data_dir, label)
    for img_file in os.listdir(label_path):
        img_path = os.path.join(label_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        X_train.append(img)
        y_train.append(int(label))
    labels.append(label)

# Create a face recognizer object and train it with the training data
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(X_train, np.array(y_train))

# Save the trained model to a file
model_file = 'model.yml'
recognizer.write(model_file)

# Save the labels to a file
with open(labels_file, 'w') as f:
    for label in labels:
        f.write(label + '\n')
