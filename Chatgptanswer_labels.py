import cv2
import numpy as np

# Load the trained face recognizer model
model_file = 'model.yml'
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_file)

# Load the labels for the trained model
labels_file = 'labels.txt'
with open(labels_file, 'r') as f:
    labels = f.read().splitlines()

# Load the test image
test_img_path = 'test2.jpg'
test_img = cv2.imread(test_img_path)

# Convert the test image to grayscale
gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)

# Loop over the detected faces
for (x, y, w, h) in faces:
    # Extract the face ROI from the grayscale image
    face_roi = gray_img[y:y+h, x:x+w]

    # Recognize the face using the trained model
    label_id, confidence = recognizer.predict(face_roi)

    # Draw a rectangle around the face and display the label and confidence
    label = labels[label_id]
    cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    print(f"{label} ({confidence:.2f})")
    cv2.putText(test_img, f"{label} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), thickness=2)

# Display the result image
resized = cv2.resize(test_img, (960, 540))
cv2.imshow('Result', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
