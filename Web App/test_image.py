import numpy as np
import cv2
from keras.models import model_from_yaml

import tkinter as tk
from tkinter import filedialog

classes = ['Asus ROG Zephyrus', 'Dell XPS', 'Huawei MateBook', 'Lenovo Ideapad', 'Macbook', 'Microsoft Surface Book']

# load YAML and create model
yaml_file = open('../model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("../model.h5")
print("Loaded model")

#file_path = 'asus_rog.jpg'
file_path = 'macbook.jpg'
#file_path = 'Microsoft_Surface_Pro_Silver.jpg'

img = cv2.imread(file_path)
img = cv2.resize(img,(170,170))
img = np.reshape(img,[1,170,170,3])

print("Running model")
predictions = loaded_model.predict(img)

print("-----------------------------------")
print(predictions[0])
print(np.argmax(predictions[0]))
print(classes[np.argmax(predictions[0])])
print("-----------------------------------")
