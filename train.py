import cv2
import os
import numpy as np

# Path to the dataset
dataset_path = 'dataset'

# Load the dataset
def load_dataset(dataset_path):
    # Initialize empty lists for images and labels
    images = []
    labels = []
    
    # Loop through each person's folder
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        
        # Loop through each image in the person's folder
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            
            # Load the image and convert to grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Convert the image to a NumPy array
            img_array = np.array(img, dtype=np.uint8)
            
            # Add the image and label to the lists
            images.append(img_array)
            labels.append(person_name)
    
    return images, labels

# Train the model
def train_model(images, labels):
    # Convert the images and labels to NumPy arrays
    images_array = np.array(images, dtype=np.uint8)
    labels_array = np.array(labels)

    # Create the face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Train the recognizer on the images and labels
    recognizer.train(images_array, labels_array)

    # Save the trained model to a YAML file
    recognizer.write('trained_model.yml')

# Load the dataset
images, labels = load_dataset(dataset_path)

# Train the model
train_model(images, labels)
