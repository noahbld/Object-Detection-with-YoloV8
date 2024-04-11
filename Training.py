!pip install ultralytics

import os
import pandas as pd
from google.colab import drive
from ultralytics import YOLO

drive.mount('/content/drive')


import os
import shutil
import random

# Chemin vers le dossier contenant les images
dossier_images = "/content/drive/My Drive/ProjetP6/DataTraining/images"
dossier_labels = "/content/drive/My Drive/ProjetP6/DataTraining/labels"
dossier_images_test = "/content/drive/My Drive/ProjetP6/DataValidation/images"
dossier_labels_test = "/content/drive/My Drive/ProjetP6/DataValidation/labels"


fichiers_images = sorted(os.listdir(dossier_images))
fichiers_labels = sorted(os.listdir(dossier_labels))

indexes = list(range(len(fichiers_images)))
random.shuffle(indexes)

nb_images_test = int(0.2* len(fichiers_images))

indexes_test = indexes[:nb_images_test]

for index in indexes_test:
    fichier_image = fichiers_images[index]
    fichier_label = fichiers_labels[index]

    chemin_source_image = os.path.join(dossier_images, fichier_image)
    chemin_destination_image = os.path.join(dossier_images_test, fichier_image)
    shutil.move(chemin_source_image, chemin_destination_image)

    chemin_source_label = os.path.join(dossier_labels, fichier_label)
    chemin_destination_label = os.path.join(dossier_labels_test, fichier_label)
    shutil.move(chemin_source_label, chemin_destination_label)



ROOT_DIR = '/content/drive/MyDrive/ProjetP6/DataProjet3'

model = YOLO("yolov8n.yaml")

results = model.train(data=os.path.join(ROOT_DIR, "config.yaml"), epochs=250) 



import shutil

source_directory = "/content/runs"
destination_directory = "/content/drive/MyDrive"


shutil.move(source_directory, destination_directory)
