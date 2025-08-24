# Rule 1: Load it using pandas (simulated as MNIST is not CSV)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Rule 2: Load and print size
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)

# Rule 3: Drop columns - simulate using DataFrame
df_train = pd.DataFrame(x_train.reshape(x_train.shape[0], -1))
df_train['label'] = y_train

# Rule 4: Selection of required columns
df_train = df_train.loc[:, list(df_train.columns[:784]) + ['label']]

# Rule 5: Identify X and Y
X = df_train.drop('label', axis=1).values
Y = df_train['label'].values
print("X shape:", X.shape)
print("Y shape:", Y.shape)

# Rule 6: Remove garbage values
df_train.dropna(inplace=True)
df_train = df_train.loc[:, (df_train != df_train.iloc[0]).any()]  # Remove constant columns

# Rule 7: Encode last column to 0 and 1 — not applicable here since it's 0-9 — keep as is
x_train = X.reshape(-1, 28, 28)
y_train_cat = to_categorical(Y, 10)

# Rule 8: Remove missing values
x_train = np.nan_to_num(x_train)
y_train_cat = np.nan_to_num(y_train_cat)
x_test = np.nan_to_num(x_test)
y_test_cat = to_categorical(y_test, 10)

# Rule 9: Bar graph of label frequencies
pd.Series(y_train).value_counts().sort_index().plot(kind='bar', color='orange')
plt.title("Frequency of Digits in Training Set")
plt.xlabel("Digit")
plt.ylabel("Count")
plt.show()

# Rule 10: Data imbalance check only (no correction needed for MNIST)
# We display distribution only, no sampling

# Rule 11: Training and testing already handled

# Rule 12: External test case (synthetic image)
custom_test = np.zeros((1, 28, 28))
x_test = np.vstack([x_test, custom_test])
y_test = np.append(y_test, [0])
y_test_cat = np.vstack([y_test_cat, to_categorical([0], 10)])

# Build the ANN model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train_cat, epochs=10, batch_size=128, validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# Training History Plots
plt.figure(figsize=(14, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Confusion Matrix
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
plt.title("Confusion Matrix - ANN on MNIST")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()

# Per-class Accuracy
print("\nPer-class Accuracy (%):")
acc_per_class = cm.diagonal() / cm.sum(axis=1)
for i, acc in enumerate(acc_per_class):
    print(f"Digit {i}: {acc * 100:.2f}%")

# Prediction Confidence Distribution
confidence_scores = np.max(y_pred_probs, axis=1)
plt.figure(figsize=(10, 5))
plt.hist(confidence_scores, bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of Model's Prediction Confidence on Test Set")
plt.xlabel('Confidence (Max Softmax Probability)')
plt.ylabel('Number of Samples')
plt.show()
