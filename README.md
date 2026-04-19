# 🐶 Dog Breed Classifier (End-to-End ML Project)

An end-to-end deep learning application that classifies dog breeds from images using a fine-tuned ResNet50 model, with an interactive Streamlit web interface for real-time predictions.

---

## 📌 Project Overview

This project demonstrates the complete machine learning lifecycle:
- Data preparation and preprocessing
- Model training using transfer learning
- Evaluation and validation
- Building an inference pipeline
- Deploying an interactive web application

The goal is to accurately classify images into one of **120 dog breeds**.

---

## 🧠 Model Details

- **Architecture:** ResNet50 (Transfer Learning)
- **Framework:** PyTorch
- **Input Size:** 224 × 224
- **Number of Classes:** 120
- **Validation Accuracy:** ~85%

The model was fine-tuned by replacing the final fully connected layer and training on a labeled dog breed dataset.

---

## ⚙️ Features

- Upload an image and predict dog breed
- Real-time inference using trained model
- Clean and interactive Streamlit UI
- Modular code structure for scalability

---

## 🏗️ Project Structure

dog-breed-classifier/
│
├── app/ # Streamlit UI
│ └── app.py
│
├── src/ # Core ML logic
│ ├── model.py
│ ├── dataset.py
│ ├── train.py
│ ├── inference.py
│
├── outputs/
│ └── models/ # Saved model + encoder
│ ├── dog_breed_model.pth
│ ├── label_encoder.pkl
│
├── config/ # Configuration files
│ └── config.yaml
│
├── notebooks/ # EDA & experiments
│ └── eda.ipynb
│
├── data/ # Dataset (optional / not included)
│
├── requirements.txt
└── README.md