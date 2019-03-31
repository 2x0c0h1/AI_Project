from keras import applications

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import (Dropout, Flatten, Dense, Conv2D,
                          Activation, MaxPooling2D)

from keras_tqdm import TQDMCallback

from keras import backend as K
if K.tensorflow_backend._get_available_gpus() == []:
    print("Not using GPU!!!")

# dimensions of our images. label = bezos, gates 2 labels
img_width, img_height = 128, 128

train_data_dir = 'training_images'
validation_data_dir = 'test_images'

epochs = 50
batch_size = 16

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3)))
model.add(Activation('relu')) #tanh
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu')) #tanh
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(96))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(6))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

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

model.fit_generator(
    train_generator,
    steps_per_epoch= 3000 // 16, # give me more data
    epochs=epochs,
    verbose=0,
    callbacks=[TQDMCallback()],
    validation_data=validation_generator,
    validation_steps= 300 // 16)
