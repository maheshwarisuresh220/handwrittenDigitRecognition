# 🖊️ Handwritten Digit Recognition — ANN with MNIST  

This repository demonstrates the implementation of a **Handwritten Digit Recognition system** using an **Artificial Neural Network (ANN)** trained on the **MNIST dataset**. The model learns to classify handwritten digits (0–9) from grayscale images and achieves **~97% accuracy** on unseen test data.  

---

## 📌 About the Project  

Handwritten digit recognition is one of the most popular beginner projects in deep learning.  
The **MNIST dataset** serves as a benchmark dataset containing **70,000 labeled digit images** (60,000 for training, 10,000 for testing). Each digit is a **28×28 grayscale image**.  

This project shows how even a **basic ANN** can achieve strong performance in image classification tasks.  

---

## 🎯 Objectives  

- ✅ Learn how to preprocess image datasets for machine learning  
- ✅ Build a simple **ANN classifier** from scratch using Keras/TensorFlow  
- ✅ Train the model and evaluate accuracy on test data  
- ✅ Visualize predictions on handwritten digits  
- ✅ Provide a reusable workflow for **students, researchers, and ML enthusiasts**  

---

## ⚙️ Features  

- **Data Preprocessing**: Normalization & one-hot encoding of labels  
- **Model Architecture**:  
  - Input Layer → Flatten (28×28 → 784 neurons)  
  - Hidden Layer 1 → 128 neurons (ReLU activation)  
  - Hidden Layer 2 → 64 neurons (ReLU activation)  
  - Output Layer → 10 neurons (Softmax activation)  
- **Training**: Adam optimizer, categorical crossentropy loss  
- **Evaluation**: Achieves ~97% accuracy on test dataset  
- **Visualization**: Displays predicted vs. actual labels for sample digits  

---

## 🚀 Getting Started  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/handwritten-digit-ann.git
cd handwritten-digit-ann
