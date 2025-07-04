🧠 Brain Tumor Detection Using Deep Learning (MRI Images)
This project focuses on building a deep learning-based system to classify brain tumors using grayscale MRI images. It leverages Convolutional Neural Networks (CNNs) and transfer learning to identify four categories of brain conditions: glioma, meningioma, pituitary tumor, and no tumor. The entire workflow includes dataset preparation, model training and comparison, evaluation, visualization, and deployment as a web application using Streamlit.

📌 Problem Statement
Brain tumors are among the most serious medical conditions, and early diagnosis can greatly impact treatment success. Manual interpretation of MRI scans is time-consuming and subject to human error. The aim of this project is to automate the classification process using deep learning to assist medical professionals in identifying tumor types accurately and efficiently.

🧠 Dataset Details
Dataset type: Grayscale MRI images in JPEG format

Categories:
  Glioma
  Meningioma
  Pituitary Tumor
  No Tumor

Folder Structure:

/Training
└── glioma
└── meningioma
└── pituitary
└── no_tumor

/Testing
└── glioma
└── meningioma
└── pituitary
└── no_tumor

Source: Publicly available dataset (Kaggle or similar platform)

🛠 Tools & Technologies Used
Python
TensorFlow, Keras
OpenCV
NumPy, Pandas
Matplotlib, Seaborn
Streamlit, ngrok

🧪 Workflow & Model Architecture
+ Data Preprocessing
+ Image resizing, grayscale conversion, normalization
+ Data augmentation to reduce overfitting

Model Development
  Compared architectures:
    + Xception
    + ResNet50
    + VGG16
    + EfficientNetB0
    + Custom CNN

Evaluation Metrics
+ Accuracy, Precision, Recall, F1-Score
+ Confusion Matrix
+ ROC-AUC Curve
Model Deployment
+ Web app built with Streamlit
+ Local deployment with ngrok tunnel

📊 Results & Analysis
Best model: EfficientNetB0
Achieved highest F1-Score and ROC-AUC
Successfully reduced overfitting with data augmentation
Web app enabled real-time predictions on user-uploaded images

