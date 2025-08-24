# LAB 10 – Even vs Odd Digit Classification (with Clean Code Best Practices)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import random

# 1. Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. Convert labels to binary: Even = 0, Odd = 1
y_train_binary = y_train % 2
y_test_binary = y_test % 2

# 4. Define the neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),     # Converts 28x28 to 784
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')     # Binary classification (even/odd)
])

# 5. Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 6. Display model architecture
model.summary()

# 7. Train the model
history = model.fit(x_train, y_train_binary,
                    epochs=10,
                    validation_split=0.2,
                    batch_size=128,
                    verbose=1)

# 8. Plot Training Accuracy & Loss
plt.figure(figsize=(12, 4))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# 9. Predict class probabilities on test set
y_pred_probs = model.predict(x_test)

# 10. Convert probabilities to binary predictions
y_pred_binary = (y_pred_probs > 0.5).astype("int32").flatten()

# 11. Confusion Matrix
cm = confusion_matrix(y_test_binary, y_pred_binary)
plt.figure(figsize=(8, 6))
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Even vs Odd)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 12. Calculate and print class-wise accuracy
acc_per_class = cm.diagonal() / cm.sum(axis=1)
labels = ['Even', 'Odd']
for i, acc in enumerate(acc_per_class):
    print(f"{labels[i]} Accuracy: {acc * 100:.2f}%")

# 13. Display 10 random test predictions
plt.figure(figsize=(15, 4))
for i in range(10):
    index = random.randint(0, len(x_test) - 1)
    img = x_test[index]
    true_label = "Odd" if y_test_binary[index] else "Even"
    pred_label = "Odd" if y_pred_binary[index] else "Even"
    confidence = y_pred_probs[index][0]

    plt.subplot(1, 10, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"P: {pred_label}\nT: {true_label}\n{confidence:.2f}")
    plt.axis('off')

plt.tight_layout()
plt.show()
