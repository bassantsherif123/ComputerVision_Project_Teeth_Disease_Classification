# 🩺 Teeth Disease Classification using Deep Learning
## 📌 Overview
This project aims to automate the diagnosis of teeth diseases classification using Convolutional Neural Networks (CNNs). It utilizes pretrained models like EfficientNetB7 and InceptionResNetV2, enhanced with data augmentation and class balancing techniques. An interactive Streamlit app is also provided for real-time predictions.

## 🚀 Features
- Classification of 7 teeth diseases
- A CNN Model built from scratch
- Fine-tuned EfficientNetB7 and InceptionResNetV2
- Interactive Streamlit interface for live prediction

## 📊 Dataset
The dataset consists of 7 distinct teeth diseases; including training, validation and testing sets.

## 🛠 Installation
```
git clone https://github.com/bassantsherif123/ComputerVision_Project_Teeth_Disease_Classification.git
```

## 🌐 Streamlit App
- Upload an image and receive instant predictions using the modified InceptionResNetV2 model. 
- The sidebar includes:
    - Class Probability pie chart
    - Class-wise prediction confidence

- If you don't have streamlit previously installed, Use:
```
pip install streamlit
```
- To run the app, Use:
```
streamlit run Deployment/Model_UI.py
```

## 📈 Results
Modified InceptionResNetV2 achieved the best accuracy with minimal overfitting with accuracies 99.77%, 98.25% and 98.15% for training, validation and testing datasets respectively; making it the best performer for this classification task