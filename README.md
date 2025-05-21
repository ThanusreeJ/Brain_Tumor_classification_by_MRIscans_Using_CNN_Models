# 🧠 Brain Tumor Classification from MRI Scans

This project is a deep learning-based web application that classifies brain MRI images into one of four categories:
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

It leverages a Convolutional Neural Network (CNN) model built using Keras and is served through a Flask web interface.




## 📊 Dataset
This project uses the Brain Tumor Classification (MRI) dataset publicly available on Kaggle, curated by Sartaj Bhuvaji.

📁 Dataset Structure
The dataset contains 3,264 labeled MRI images categorized into four classes:

- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

Images are grayscale and vary in size but are typically resized to 224×224 pixels during preprocessing.

📌 Distribution (Approximate)
Tumor Type	Number of Images
Glioma Tumor	926
Meningioma Tumor	937
Pituitary Tumor	901
No Tumor	500

🧪 Data Use
All images are preprocessed and normalized before being passed to the model.

The dataset is split into training, validation, and test sets using an 80-10-10 or similar split.

Augmentation techniques (like rotation, flipping, zooming) may be used to improve model generalization.




## ⚙️ Setup Instructions
1. Clone the Repository
```bash
git clone https://github.com/SandeepBalimidi/Brain_tumor_classification.git
cd Brain_tumor_classification
```
2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```
4. Run the App
```bash
python app.py
```
Visit http://127.0.0.1:5000/ in your browser.

## 🧠 Model Info
Framework: TensorFlow / Keras

Model Architecture: Pre-trained Xception model with custom classification head (Transfer Learning)

Input Size: 224 × 224 pixels

Dataset: Kaggle Brain Tumor MRI Classification

📈 Accuracy
Metric	Value
Training Acc	~98%
Validation Acc	~95%

🔍 Note: Model performance may vary depending on image preprocessing, augmentation strategies, and dataset split ratios.

## 📊 Future Improvements
🔬 Integrate Grad-CAM for tumor localization and model interpretability

📱 Convert model to TensorFlow Lite for mobile or edge deployment

☁️ Deploy the application on cloud platforms like Heroku, Render, or AWS for public access
