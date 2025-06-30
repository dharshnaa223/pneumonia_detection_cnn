# TITLE:A Deep Learning Approach to Pneumonia Diagnosis via Image Classification

This project presents an end-to-end deep learning pipeline to detect **Pneumonia** from chest X-ray images using multiple CNN architectures. From data preprocessing and model training to evaluation and deployment using Streamlit, the solution showcases how artificial intelligence can support faster and more accurate medical diagnosis.


## 🎯 Project Objective

- Develop a deep learning-based image classification model for early detection of pneumonia.
- Compare the performance of four CNN-based models: **Basic CNN**, **VGG19**, **ResNet50**, and **Xception**.
- Handle real-world dataset challenges such as **class imbalance** and **data variability**.
- Deploy the best-performing model in an accessible **Streamlit** web app for real-time X-ray diagnosis.


## 📊 Dataset Description

- **Source**: [Kaggle – Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total Images**: ~5,863 X-rays  
- **Classes**: `PNEUMONIA` and `NORMAL`
- **Structure**:
  - Already split into `train`, `validation`, and `test` folders
  - Contains grayscale chest X-ray images of various sizes and orientations



## 🧹 Data Preprocessing

- **Resizing**: All images resized to 224x224 pixels for consistency across all models.
- **Normalization**: Pixel values rescaled to the range [0, 1].
- **Data Augmentation**: Applied on training data using `ImageDataGenerator` to increase generalization:
  - Rotation
  - Horizontal Flip
  - Width and Height Shift
  - Zoom
- **Class Imbalance Handling**:
  - Applied **oversampling** of the minority class (NORMAL) during training.
  - Ensured balanced batches for training and validation.
  
These steps were implemented in TensorFlow/Keras to prepare the data for robust model performance.


## 🧠 Deep Learning Models Evaluated

I trained and compared the following four models:

### 🔸 1. **Basic CNN**
- Built from scratch with 3 convolutional layers + ReLU, MaxPooling, Dropout, and Dense layers
- Simple architecture used as a baseline
- **Test Accuracy:** 83%

### 🔸 2. **VGG19** *(Deployed Model – Best Accuracy)*
- Pretrained on ImageNet, with custom classification layers added
- Fine-tuned last few layers to adapt to medical imaging
- Deep architecture enabled high feature extraction capability
- **Test Accuracy:** ⭐ **94%**

### 🔸 3. **ResNet50**
- Leveraged deep residual blocks to mitigate vanishing gradients
- Used transfer learning with ImageNet weights and fine-tuned layers
- **Test Accuracy:** 92%

### 🔸 4. **Xception**
- Lightweight and efficient, based on depthwise separable convolutions
- Faster inference, slightly less accurate
- **Test Accuracy:** 81%

---

## 📈 Model Evaluation Metrics

Each model was evaluated using the following:

- ✅ **Accuracy**
- ✅ **Precision, Recall, F1-score**
- ✅ **Confusion Matrix**
- ✅ **Training vs. Validation Loss and Accuracy Curves**

Performance plots and matrices were generated in the Jupyter notebook using **Matplotlib** and **Seaborn**.



## 🧪 Final Model Selection & Justification

The **VGG19 model** achieved the highest classification accuracy of **94%**, outperforming other models in both recall and F1-score. Its deep architecture allowed for better generalization and robust feature extraction, resulting in fewer false positives and false negatives.

This model was selected for deployment due to its:

- Highest accuracy
- Consistent validation performance
- Lower misclassification rate
- Ease of transfer learning integration

---

## 🚀 Streamlit Web App Deployment

To make the model accessible and user-friendly, I developed a **Streamlit** app with the following features:

- 📤 Upload chest X-ray images
- 🧠 Use trained VGG19 model for prediction
- 📊 Display predicted class (`Pneumonia` / `Normal`) with confidence score
- 📷 Easy-to-use interface for real-time inference

### ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```



### Download Trained Models:

All four trained models used in this project are available in a single Google Drive folder:

📁 **[Click here to download trained models (.h5)](https://drive.google.com/drive/folders/11KlSGEc-jJOcnWf1NKM8cb0lRc8AJhsA?usp=sharing)**

### 📌 Included Models:
- 🧠 Basic CNN (`basic_cnn_model(1).h5`)
- 🏗️ VGG19 (`base_vgg19.h5`)
- 🧠 ResNet50 – Best Accuracy (`final_fine_tuned_resnet.h5`)
- ⚡ Xception (`base_xception.h5`)


## ✅ Conclusion

This project demonstrates how deep learning can be effectively applied to medical image classification, specifically for detecting pneumonia from chest X-ray images. Through the use of convolutional neural networks and transfer learning techniques, I built multiple models capable of distinguishing between normal and pneumonia cases with high confidence.

Beyond model development, I focused on addressing real-world challenges such as class imbalance, overfitting, and limited labeled data. By incorporating data augmentation, model regularization, and evaluation using multiple metrics, the models were made more robust and generalizable.

To ensure practical usability, the best-performing model was deployed via a **Streamlit web application**, allowing real-time prediction on user-uploaded X-rays. This bridges the gap between research and real-world deployment, making the tool accessible to both medical professionals and end users.

Overall, this project highlights the power of combining deep learning with accessible deployment tools to build meaningful, real-world AI solutions for healthcare.



