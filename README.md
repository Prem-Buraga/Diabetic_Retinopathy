This repository contains code and resources for the APTOS 2019 Blindness Detection competition on Kaggle. The goal of this competition is to detect diabetic retinopathy in retinal images. This project includes data preprocessing, exploratory data analysis (EDA), model training using deep learning, and performance evaluation.

Table of Contents
Dataset
Installation
Exploratory Data Analysis
Data Preprocessing
Modeling
Deep Convolutional Neural Network (DCNN)
AlexNet
Training
Evaluation
Results
Contributing
License
Dataset
The dataset can be downloaded from the Kaggle APTOS 2019 Blindness Detection competition page. Extract the dataset and place it in the Dataset directory.

Installation
To run the code in this repository, you need to have the following libraries installed:

numpy
pandas
seaborn
matplotlib
OpenCV
TensorFlow
scikit-learn
scikit-plot
You can install the required packages using the following command:

sh
Copy code
pip install numpy pandas seaborn matplotlib opencv-python tensorflow scikit-learn scikit-plot
Exploratory Data Analysis
The EDA includes loading the data, checking for class imbalances, and visualizing the distribution of the target variable.

Data Preprocessing
Data preprocessing involves augmenting the minority classes to balance the dataset and preparing the images for model input. The images are resized and normalized.

Modeling
Deep Convolutional Neural Network (DCNN)
A custom DCNN is built for the classification task. The model architecture consists of multiple convolutional and max-pooling layers followed by dense layers.

AlexNet
The AlexNet architecture is also implemented for comparison. It consists of multiple convolutional layers followed by fully connected layers.

Training
The models are trained using the prepared dataset. The training process involves splitting the data into training and validation sets and then fitting the model on the training data.

Evaluation
The performance of the models is evaluated using accuracy and loss metrics. Confusion matrices are plotted to visualize the classification performance.

Results
The final accuracy and loss plots for both training and validation sets are visualized to assess the model's performance. The classification report provides a detailed performance evaluation.

Final Accuracy of emotion after 6 epochs: 97.43%
