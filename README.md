🧠 Brain Tumor Classification using Deep Learning & Streamlit
An end-to-end Deep Learning project that classifies brain MRI images into four tumor types — built with TensorFlow, Keras, and Streamlit.
This project demonstrates both model development (CNN + Transfer Learning) and interactive deployment as a web app.

🚀 Project Overview
This project aims to automate brain tumor detection using MRI images.
It leverages Convolutional Neural Networks (CNNs) and Transfer Learning models like ResNet50, MobileNetV2, and EfficientNetB0 for high accuracy and generalization.

The trained model is deployed as a Streamlit web application, allowing users to upload MRI images and instantly view:

✅ Predicted tumor type
📊 Confidence scores
🔥 Grad-CAM heatmap (to visualize model attention)
📂 Dataset
The dataset contains MRI scans categorized into 4 classes:

Tumor Type	Description
Glioma	Tumor arising from glial cells in the brain or spine
Meningioma	Tumor originating from the meninges
Pituitary	Tumor of the pituitary gland
No Tumor	Normal MRI scans without tumor presence
Structure after extraction:

🧩 Project Workflow
1️⃣ Understand the Dataset
Inspect image samples and class distribution
Check for class imbalance
Visualize random MRI examples
2️⃣ Data Preprocessing
Normalize pixel values to [0,1]
Resize all images to 224×224
Force RGB channels for grayscale MRI scans
3️⃣ Data Augmentation
Random rotations, flips, zooms, brightness, and shifts
Helps prevent overfitting and improve robustness
4️⃣ Model Building
🧠 Custom CNN
Built from scratch using Conv2D, BatchNorm, Dropout layers
Tuned for high accuracy with minimal overfitting
⚙️ Transfer Learning
Used multiple pretrained ImageNet models:

ResNet50
MobileNetV2
EfficientNetB0
Fine-tuned the top layers and replaced the classifier for 4 tumor categories.

5️⃣ Model Training
Optimizer: Adam
Loss: SparseCategoricalCrossentropy
Metrics: Accuracy
Callbacks:
EarlyStopping – prevent overfitting
ModelCheckpoint – save best model automatically
6️⃣ Evaluation
Evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
Classification Report
7️⃣ Model Comparison
Compared Custom CNN vs Pretrained Models and selected the best based on weighted F1-score.

📊 Results Summary
Model	Accuracy	Precision	Recall	F1-score
Custom CNN	0.93	0.92	0.93	0.92
ResNet50	0.95	0.95	0.95	0.95
MobileNetV2	0.96	0.96	0.96	0.96
EfficientNetB0 (Best)	0.97	0.97	0.97	0.97
🌐 Streamlit Application
🎯 Features
Upload MRI images (.jpg, .jpeg, .png)
Choose model (Custom CNN / ResNet / MobileNet / EfficientNet)
See tumor prediction with confidence scores
Visualize Grad-CAM heatmap showing model’s focus area
