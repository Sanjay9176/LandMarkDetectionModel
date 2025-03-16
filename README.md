Landmark Classification using VGG19

Overview

This project is a deep learning-based classification model using the VGG19 architecture to classify landmarks from images. The dataset is preprocessed and trained using TensorFlow and Keras.

Features

Custom Dataset Handling: Reads and preprocesses images from a structured dataset.

Label Encoding: Uses LabelEncoder to convert categorical labels to numerical values.

Data Augmentation: Uses ImageDataGenerator for improved generalization.

Pretrained Model (VGG19): Fine-tuned to classify landmarks.

Custom Training Loop: Uses GradientTape for manual optimization.

Installation

To set up the project, clone the repository and install the required dependencies:

pip install tensorflow pandas numpy opencv-python matplotlib scikit-learn pillow

Usage

Run the main script to preprocess data and train the model:

python train.py

Folder Structure

├── train.csv              # CSV file containing image metadata
├── photos/                # Folder containing landmark images
├── Model.h5               # Saved trained model
├── README.md              # Project documentation
└── train.py               # Main script for training and evaluation

Areas for Improvement

1. Optimize Image Loading

Current Code:

def get_image_from_numbers(num, df):
    fname, label = df.iloc[num, :]
    folder = base_path + "0/0/" + fname

Optimization: Instead of hardcoded folder paths, dynamically construct paths using os.path.join().

folder = os.path.join(base_path, fname[0], fname[1], fname[2], fname)

2. Batch Processing with Generators

Current Code:

for it in range(int(np.ceil(len(train) / batch_size))):
    x_train, y_train = get_batch(train, it * batch_size, batch_size)

Optimization: Use ImageDataGenerator.flow_from_dataframe() instead of manual batch processing for better efficiency.

train_datagen.flow_from_dataframe(train, directory=base_path, target_size=(224,224), batch_size=batch_size)

3. Improve Model Performance with Transfer Learning

Current Code:

source_model = VGG19(weights=None)

Optimization: Use pretrained ImageNet weights to improve accuracy and reduce training time.

source_model = VGG19(weights='imagenet', include_top=False)

4. Use ModelCheckpoint for Best Model Saving

Current Code:

model.save("Model.h5")

Optimization: Instead of saving at the end, save only the best model using:

checkpoint = keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_accuracy')

5. Replace Manual Training with model.fit()

Current Code:

for e in range(epochs):
    for it in range(int(np.ceil(len(train) / batch_size))):
        train_step(x_train, y_train)

Optimization: Replace with standard Keras training for better efficiency.

model.fit(train_generator, validation_data=val_generator, epochs=epochs, callbacks=[checkpoint])

Contributing

Feel free to contribute to this project by optimizing the code further or adding new features!
