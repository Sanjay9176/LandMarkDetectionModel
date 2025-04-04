# Landmark Image Classification using VGG19

## Overview
This project focuses on classifying landmark images using a deep learning model based on VGG19. It processes a dataset of landmarks, extracts features, and trains a neural network to recognize different landmark categories. The model is built using TensorFlow and Keras, incorporating data augmentation and image preprocessing techniques.

## Features
- Loads and preprocesses landmark images automatically
- Uses data augmentation to improve model generalization
- Implements the pre-trained VGG19 model for feature extraction
- Trains a neural network to classify different landmarks
- Evaluates the model and visualizes training progress
- Provides a batch-processing mechanism for efficient training

## Installation
Before running the code, install the required Python packages:
```bash
pip install pandas numpy tensorflow scikit-learn opencv-python matplotlib pillow
```

## Dataset Setup
Ensure the dataset is in a CSV file named `train.csv`, where:
- `id` corresponds to the image filename.
- `landmark_id` represents the label.
- Images are stored in a structured folder `./photos/`.

## How to Run
### 1. Import necessary libraries
```python
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
```

### 2. Load and filter dataset
```python
df = pd.read_csv("train.csv")
df = df.loc[df["id"].str.startswith(('00'), na=False), :]
num_classes = len(df["landmark_id"].unique())
num_data = len(df)
print("THE NUMBER OF NUM_CLASSES :", num_classes)
print("THE NUMBER OF NUM_DATA :", num_data)
```

### 3. Encode labels
```python
lencoder = LabelEncoder()
lencoder.fit(df["landmark_id"])
def encode_label(label):
    return lencoder.transform(label)
def decode_label(label):
    return lencoder.inverse_transform(label)
```

### 4. Load images and visualize samples
```python
base_path = "./photos/"
fig = plt.figure(figsize=(25, 25))
for i in range(1, 9):
    ri = random.choices(os.listdir(base_path), k=3)
    folder = base_path + "0/0/" + ri[2]
    if not os.path.exists(folder):
        continue
    files_in_folder = os.listdir(folder)
    if not files_in_folder:
        continue
    random_img = random.choice(files_in_folder)
    img_path = os.path.join(folder, random_img)
    img = np.array(Image.open(img_path))
    fig.add_subplot(1, 8, i)
    plt.imshow(img)
    plt.axis("off")
plt.show()
```

### 5. Define model architecture
```python
learning_rate = 0.0001
loss_function = "sparse_categorical_crossentropy"
source_model = VGG19(weights=None)
model = Sequential()
for layer in source_model.layers[:-1]:
    model.add(layer)
model.add(Dense(num_classes, activation="softmax"))
model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss=loss_function, metrics=["accuracy"])
```

### 6. Train the model
```python
train, val = np.split(df.sample(frac=1), [int(0.8 * len(df))])
batch_size = 16
epochs = 1
for e in range(epochs):
    print("Epoch:", e + 1, "/", epochs)
    for it in range(int(np.ceil(len(train) / batch_size))):
        x_train, y_train = get_batch(train, it * batch_size, batch_size)
        train_step(x_train, y_train)
model.save("Model.h5")
```

### 7. Evaluate the model
```python
good_preds, bad_preds = [], []
for it in range(int(np.ceil(len(val) / batch_size))):
    x_val, y_val = get_batch(val, it * batch_size, batch_size)
    result = model.predict(x_val)
    for idx, res in enumerate(result):
        if np.argmax(res) != y_val[idx]:
            bad_preds.append([idx, np.argmax(res), res[np.argmax(res)]])
        else:
            good_preds.append([idx, np.argmax(res), res[np.argmax(res)]])
```

## Improvements
1. **Data Augmentation** - Use `ImageDataGenerator` to apply transformations like rotation, flipping, and zoom to enhance model generalization.
2. **Transfer Learning** - Load pre-trained weights from VGG19 instead of training from scratch for better accuracy.
3. **Batch Normalization** - Add `BatchNormalization()` layers in between to stabilize learning and improve convergence speed.
4. **Hyperparameter Tuning** - Experiment with different optimizers (Adam, SGD) and learning rates to improve performance.
5. **Efficient Image Loading** - Instead of loading all images into memory, use `tf.data.Dataset` for efficient image processing.

##Impoertant Note

## Author
[Sanjay Kumar Purohit]
