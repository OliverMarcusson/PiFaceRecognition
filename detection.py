import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained model
recognizer.read('path/to/trained_model.yml')

# Start the camera
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    ret, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through each face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Extract the face ROI
        roi_gray = gray[y:y+h, x:x+w]

        # Recognize the face
        id_, confidence = recognizer.predict(roi_gray)

        # Display the name and confidence level
        if confidence < 70:
            # Replace '1' with the ID of the person you want to recognize
            if id_ == 1:
                name = 'John Doe'
            else:
                name = 'Unknown'

            cv2.putText(img, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, f'Confidence: {confidence}', (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the output
    cv2.imshow('Face Recognition', img)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()