import os

# Make a script that renames all the files in a directory to an integer starting from 1
# Path to the dataset
dataset_path = 'dataset'

for person in os.listdir(dataset_path):
    i = 1
    for img in os.scandir(os.path.join(dataset_path, person)):
        os.rename(img.path, os.path.join(dataset_path, person, f'{i}.jpg'))
        i += 1