# -*- coding: utf-8 -*-
"""MobileNetV2_Optimization_Embedded.ipynb

MobileNetV2 Optimization for Embedded Computer Vision
-----------------------------------------------------
This script demonstrates how to optimize a MobileNetV2 model using transfer learning,
fine-tuning, and FP16 quantization for deployment on resource-constrained embedded systems.

Author: Youssef Ennouri
Date: March 2025
"""

import os
import time
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split

# Import TensorFlow Model Optimization toolkit
import tensorflow_model_optimization as tfmot

# Enable mixed precision training to reduce memory consumption
mixed_precision.set_global_policy('mixed_float16')

# Load CIFAR-10 dataset
(x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

# Split data into training and validation sets (80% train, 20% validation)
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=0.2, random_state=42
)

print(f"Train set: {x_train.shape[0]} images")
print(f"Validation set: {x_val.shape[0]} images")
print(f"Test set: {x_test.shape[0]} images")

def preprocess_image(img, training=False):
    """
    Preprocess images for model training and inference.
    Includes resizing, data type conversion, and data augmentation during training.
    
    Args:
        img: Input image to preprocess
        training: Boolean flag to enable data augmentation
        
    Returns:
        Preprocessed image in float16 format
    """
    img = tf.image.resize(img, (128, 128))
    img = tf.cast(img, tf.uint8)  # Keep as uint8 to conserve RAM
    
    # Apply data augmentation during training
    if training:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.8, 1.2)
    
    # Apply MobileNetV2 preprocessing and convert to float16
    img = preprocess_input(tf.cast(img, tf.float16))
    return img

# Create efficient TensorFlow datasets
batch_size = 16

# Training dataset with data augmentation
train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .map(lambda img, label: (preprocess_image(img, training=True), tf.one_hot(label[0], 10)))
    .shuffle(1000)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

# Validation dataset
val_dataset = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .map(lambda img, label: (preprocess_image(img, training=False), tf.one_hot(label[0], 10)))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

# Test dataset
test_dataset = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .map(lambda img, label: (preprocess_image(img, training=False), tf.one_hot(label[0], 10)))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

# Build model with MobileNetV2 base
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dropout(0.3)(x)  # Add dropout to reduce overfitting
x = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=x)

# Freeze base model layers for initial training
for layer in base_model.layers:
    layer.trainable = False

# Define callbacks for training optimization
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='transfer_learning_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

tf.keras.backend.clear_session()
gc.collect()

# Phase 1: Train only the classification layer
print("Phase 1: Training only the classification layers")
history1 = model.fit(
    train_dataset,
    epochs=20,
    validation_data=val_dataset,
    callbacks=callbacks
)

tf.keras.backend.clear_session()
gc.collect()

# Phase 2: Fine-tuning the last layers of the base model
print("Phase 2: Fine-tuning the last layers")
# Unfreeze the last 20 layers of the base model
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

tf.keras.backend.clear_session()
gc.collect()

# Train with fine-tuning
history2 = model.fit(
    train_dataset,
    epochs=20,
    validation_data=val_dataset,
    callbacks=callbacks
)

tf.keras.backend.clear_session()
gc.collect()

# Load the best saved model
model = tf.keras.models.load_model('transfer_learning_model.h5')

# Evaluate model on test set
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Create TFLite converter with FP16 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model_fp16 = converter.convert()

# Save the optimized TFLite model
with open("model_fp16.tflite", "wb") as f:
    f.write(tflite_model_fp16)

# Load the TFLite model for evaluation
interpreter = tf.lite.Interpreter(model_path="model_fp16.tflite")
interpreter.allocate_tensors()

# Get input and output tensor indices
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_tflite(interpreter, dataset):
    """
    Make predictions using the TFLite model and calculate accuracy.
    
    Args:
        interpreter: TFLite interpreter
        dataset: Dataset to evaluate on
        
    Returns:
        Accuracy of the TFLite model
    """
    correct = 0
    total = 0

    for images, labels in dataset:
        for i in range(len(images)):
            input_data = np.expand_dims(images[i], axis=0).astype(np.float32)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

            pred_label = np.argmax(output)
            true_label = np.argmax(labels[i])

            if pred_label == true_label:
                correct += 1
            total += 1

    return correct / total

def get_model_size(file_path):
    """
    Calculate file size in bytes, KB, and MB.
    
    Args:
        file_path: Path to the model file
        
    Returns:
        Tuple of (size_bytes, size_kb, size_mb)
    """
    size_bytes = os.path.getsize(file_path)
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024
    return size_bytes, size_kb, size_mb

# Model file paths
keras_model_path = "transfer_learning_model.h5"
tflite_model_path = "model_fp16.tflite"

# Get model sizes
keras_size = get_model_size(keras_model_path)
tflite_size = get_model_size(tflite_model_path)

# Display size comparison results
print("\n--- Model Size Comparison ---")
print(f"📌 Keras Model (.h5): {keras_size[2]:.2f} MB ({keras_size[0]:,} bytes)")
print(f"📌 TFLite Model (.tflite): {tflite_size[2]:.2f} MB ({tflite_size[0]:,} bytes)")

# Calculate size reduction percentage
size_reduction = ((keras_size[0] - tflite_size[0]) / keras_size[0]) * 100
print(f"💾 Size Reduction: {size_reduction:.2f}%")

# Calculate and display accuracy comparison
loss, accuracy = model.evaluate(test_dataset)
print(f"📌 Keras Model Accuracy (.h5): {accuracy * 100:.2f}%")
tflite_accuracy = predict_tflite(interpreter, test_dataset)
print(f"📌 TFLite Model Accuracy (.tflite): {tflite_accuracy * 100:.2f}%")