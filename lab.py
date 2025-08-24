# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Build the ANN model
model = Sequential([
    Flatten(input_shape=(28, 28)),          # Flatten 28x28 images into 784-length vector
    Dense(128, activation='relu'),          # Hidden layer with 128 neurons and ReLU activation
    Dense(64, activation='relu'),           # Hidden layer with 64 neurons and ReLU activation
    Dense(10, activation='softmax')         # Output layer with 10 neurons (one per digit), softmax activation
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train_cat, epochs=10, batch_size=128, validation_split=0.1)

# Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# Plot Training Loss and Accuracy
plt.figure(figsize=(14,5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Confusion Matrix & Visualization
# Predict labels for test set
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix heatmap
plt.figure(figsize=(10, 7))
plt.title("Confusion Matrix - ANN on MNIST")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()

# Per Class Accuracy
print("\nPer-class Accuracy (%):")
class_wise_accuracy = cm.diagonal() / cm.sum(axis=1)
for i, acc in enumerate(class_wise_accuracy):
    print(f"Digit {i}: {acc*100:.2f}%")

# Prediction Confidence Distribution
# Take the max probability from softmax output as confidence
confidence_scores = np.max(y_pred_probs, axis=1)

plt.figure(figsize=(10,5))
plt.hist(confidence_scores, bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of Model's Prediction Confidence on Test Set")
plt.xlabel('Confidence (Max Softmax Probability)')
plt.ylabel('Number of Samples')
plt.show()
