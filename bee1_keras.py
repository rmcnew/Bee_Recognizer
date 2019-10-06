from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

# Paths to datasets
current_dir = os.getcwd()
datasets_dir = os.path.join(current_dir, 'datasets')
bee_dir = os.path.join(datasets_dir, 'BEE1')

training_dir = os.path.join(bee_dir, 'training')
testing_dir = os.path.join(bee_dir, 'testing')
validation_dir = os.path.join(bee_dir, 'validation') 

bee_training_dir = os.path.join(training_dir, 'bee')
no_bee_training_dir = os.path.join(training_dir, 'no_bee')

bee_testing_dir = os.path.join(testing_dir, 'bee')
no_bee_testing_dir = os.path.join(testing_dir, 'no_bee')

bee_validation_dir = os.path.join(validation_dir, 'bee')
no_bee_validation_dir = os.path.join(validation_dir, 'no_bee')

# constants 
image_width = 32
image_height = 32

epochs = 15
batch_size = 128

# count images
num_bee_tr = len(os.listdir(bee_training_dir))
num_no_bee_tr = len(os.listdir(no_bee_training_dir))

num_bee_val = len(os.listdir(bee_validation_dir))
num_no_bee_val = len(os.listdir(no_bee_validation_dogs_dir))

total_train = num_bee_tr + num_no_bee_tr
total_val = num_bee_val + num_no_bee_tr

# convert images 
train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(image_height, image_width),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(image_height, image_width),
                                                              class_mode='binary')


model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(image_width, image_height ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


