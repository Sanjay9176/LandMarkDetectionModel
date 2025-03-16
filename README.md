Landmark Image Classification using VGG19

Overview

This project focuses on classifying landmark images using a deep learning model based on VGG19. The goal is to preprocess the dataset, train a neural network, and evaluate its performance. The model uses TensorFlow and Keras, employing data augmentation and feature extraction techniques to improve accuracy.

Features

Automatically loads and preprocesses images from a dataset.

Implements data augmentation for better generalization.

Utilizes the pre-trained VGG19 model for feature extraction.

Trains a neural network to classify landmark images.

Evaluates the model and visualizes accuracy/loss trends.

Installation

Before running the code, install the required Python packages:

pip install pandas numpy tensorflow scikit-learn matplotlib opencv-python pillow

Dataset Setup

Ensure the CSV file (train.csv) is present in the working directory.

Place landmark images inside the photos/ directory, structured in a hierarchical format.

How to Run

1. Import necessary libraries

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder
import cv2
import os
import random
from matplotlib import pyplot as plt
from PIL import Image

2. Load and preprocess the dataset

df = pd.read_csv("train.csv")
df = df[df["id"].str.startswith(('00'), na=False)]

3. Encode landmark labels

lencoder = LabelEncoder()
lencoder.fit(df["landmark_id"])
def encode_label(label):
    return lencoder.transform(label)
def decode_label(label):
    return lencoder.inverse_transform(label)

4. Define function to retrieve images

def get_image_from_numbers(num, df):
    fname, label = df.iloc[num, :]
    full_path = f"./photos/{fname}.jpg"
    im = cv2.imread(full_path)
    if im is None:
        print("Error loading image:", full_path)
        return None, None
    return im, label

5. Build the deep learning model

learning_rate = 0.0001
loss_function = "sparse_categorical_crossentropy"
source_model = VGG19(weights=None)
model = Sequential()
for layer in source_model.layers[:-1]:
    model.add(layer)
model.add(Dense(len(df["landmark_id"].unique()), activation="softmax"))
model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss=loss_function, metrics=["accuracy"])

6. Train the model

batch_size = 16
epochs = 10
train, val = np.split(df.sample(frac=1), [int(0.8 * len(df))])
for e in range(epochs):
    print(f"Epoch {e + 1}/{epochs}")
    for it in range(int(np.ceil(len(train) / batch_size))):
        x_train, y_train = get_batch(train, it * batch_size, batch_size)
        model.fit(x_train, y_train, epochs=1, verbose=1)

7. Save and evaluate the model

model.save("Model.h5")
model.evaluate(x_test, y_test)

Results

The model is trained for multiple epochs and achieves reasonable accuracy. You can further improve the model by:

Fine-tuning VGG19 layers

Experimenting with different optimizers and learning rates

Implementing data augmentation techniques

Improvements

Optimize Data Loading: Use tf.data.Dataset instead of manual loading for efficiency.

Better Augmentation: Implement Keras ImageDataGenerator to augment training data.

Fine-Tuning VGG19: Allow some layers to be trainable for improved accuracy.

Batch Processing: Utilize NumPy arrays instead of lists for faster processing.

Author

Sanjay Kumar Purohit

