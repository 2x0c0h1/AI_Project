import numpy as np
import cv2
from keras.models import model_from_yaml

import tkinter as tk
from tkinter import filedialog

# load YAML and create model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model")

file_path = 'asus_rog.jpg'

img = cv2.imread(file_path)
img = cv2.resize(img,(128,128))
img = np.reshape(img,[1,128,128,3])

print("Running model")

classes = loaded_model.predict_classes(img)
print("Predicted : ", classes)
