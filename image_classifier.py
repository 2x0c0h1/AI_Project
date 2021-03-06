from keras import applications, optimizers, regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import (Dropout, Flatten, Dense, Conv2D,
                          Activation, MaxPooling2D, BatchNormalization)
from keras_tqdm import TQDMCallback

#Image Dimensions
img_width, img_height, img_depth = 128, 128, 3

#Image folder directories
train_data_dir = 'images'
validation_data_dir = 'test_images'

#Building Model
model = Sequential()

#Input
model.add(Conv2D(32, (3, 3), input_shape=(img_width, img_height, img_depth)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(96, (3, 3)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(activation='relu', units=img_width))
model.add(BatchNormalization())

#Output
model.add(Dense(6, activation='softmax'))

#adadelta = optimizers.Adadelta(lr=0.98, rho=0.95, epsilon=None, decay=0.0)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.3,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

epochs = 40
batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical', shuffle=True)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

H = model.fit_generator(
    train_generator,
    steps_per_epoch= 2111 // batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=230 // batch_size)

import matplotlib.pyplot as plt
import numpy as np

plt.figure()
N = epochs
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(np.arange(0, N), H.history["loss"], label="train_loss")
axarr[0].plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
axarr[1].plot(np.arange(0, N), H.history["acc"], label="train_acc")
axarr[1].plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.tight_layout()
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.show()

# serialize model to YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
