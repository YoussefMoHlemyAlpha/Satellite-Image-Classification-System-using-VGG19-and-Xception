# 🌍 Satellite Image Classification using CNN, VGG19, and Xception
![image](https://github.com/user-attachments/assets/ee22ae67-cbb8-4dae-80c6-8401fe51b93e)
### This project classifies satellite images into four categories: Cloudy, Desert, Green Area, and Water. It includes models such as a custom CNN, VGG19, and Xception, implemented using TensorFlow, and is deployed as a web app using Streamlit.

## 📂 Dataset
The dataset is downloaded from Kaggle, containing satellite images of the following classes:
https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification
Cloudy
Desert
Green Area
Water

### Dataset Preprocessing
Images resized to 80x80 pixels.
Dataset split into training (85%) and testing (15%) subsets.
Normalized image pixel values (dividing by 255).

### 🛠️ Libraries and Tools
Python
TensorFlow & Keras
OpenCV
Matplotlib, Seaborn (for plotting)
scikit-learn (for classification report, confusion matrix)
Streamlit (for web app deployment)

### 📊 Models
1. Custom CNN
A basic CNN architecture with L2 regularization and Dropout to avoid overfitting:-
Conv2D layers with 32, 64, and 128 filters.
MaxPooling2D for down-sampling.
Dense layers with 256 neurons and ReLU activation.
Output layer with 4 neurons and softmax activation.

3. VGG19
A pre-trained VGG19 model with custom dense layers and frozen base layers for initial training.

4. Xception
A pre-trained Xception model with custom classification layers, using transfer learning.

## 📊 Model Performance

| Model       | Training Accuracy | Testing Accuracy | Training Loss | Testing Loss |
|-------------|-------------------|------------------|---------------|--------------|
| Custom CNN  |      70%          |          73%     |      0.6419   |       0.5496 |
| VGG19       |      95%          |          96%     |      0.1150   |       0.0965 |
| Xception    |      99%          |          94%     |      0.0410   |       0.3511 |


### 🚀 Streamlit Web App
You can interact with the model through a web app hosted using Streamlit, where users can upload satellite images and get real-time predictions.

### 📝 Conclusion
This project showcases CNN-based models for satellite image classification, with the option to deploy as a user-friendly web app using Streamlit.

### 📧 Contact
Reach out at youssefhelmy4444@gmail.com for any questions or suggestions.

