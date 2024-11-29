# Autism Detection Using ResNet50 & Xception Transfer Learning

This project demonstrates the use of advanced Deep Learning techniques to detect autism in individuals based on images using ResNet50 and Xception models. The approach leverages Transfer Learning, which utilizes pre-trained models on large datasets like ImageNet and fine-tunes them for specific tasks like autism detection.

The system provides an easy-to-use graphical interface (GUI) built using Tkinter for interacting with the models. Users can upload datasets, preprocess them, train models, and visualize results. Additionally, users can make predictions on new images using the trained models.

# Key Features
Dataset Upload: Upload a folder containing images for training and testing the models.

Data Preprocessing: Resize images, normalize pixel values, and split the dataset into training and test sets.

Model Training: Train two deep learning models (ResNet50 and Xception) using Transfer Learning techniques.

Model Evaluation: View performance metrics like Accuracy, Precision, Recall, and F1-Score for both models.

Confusion Matrix: Visualize the confusion matrix to analyze model classification performance.

Comparison Graph: Generate a comparative bar graph to visualize model performance across different metrics.
Prediction: Make real-time predictions on test images and classify autism-related images.
# Technologies Used:
# 1. Deep Learning (Transfer Learning)
ResNet50: A powerful pre-trained model from Keras that is used for feature extraction and transfer learning. It uses residual connections to enable training of deeper networks.
Xception: Another pre-trained model from Keras, similar to ResNet but with depthwise separable convolutions for efficient learning.

# 2. Keras:
Keras is the primary library for building the CNN models. It provides simple interfaces for creating, training, and evaluating deep learning models.
# 3. TensorFlow:
TensorFlow is the backend engine for Keras, enabling high-performance training of models.
# 4. OpenCV:
OpenCV is used for image loading, processing, and visualization. It provides the necessary functions to handle and manipulate images (resizing, normalizing, etc.).
# 5. Matplotlib & Seaborn:
Used for data visualization. Matplotlib generates the graphs and plots, while Seaborn is used for heatmaps (Confusion Matrix visualization).
# 6. Scikit-learn:
Scikit-learn is used for evaluation metrics such as accuracy, precision, recall, f1-score, and confusion matrix calculations.
# 7. Tkinter:
The GUI framework used to create the user interface for interacting with the model. It allows users to upload datasets, process data, train models, and view results.
# 8. NumPy & Pandas:
NumPy is used for data manipulation (arrays and matrices), while Pandas is used for data processing and handling structured data like performance metrics in tabular form.
# 9. ModelCheckpoint:
This Keras callback is used to save the best model during training based on validation loss, ensuring that the model with the best performance is retained.

# Requirements
The following Python libraries are required for this project:

Python 3.x

Tkinter (GUI)

OpenCV (pip install opencv-python)

Keras (pip install keras)

TensorFlow (pip install tensorflow)

Matplotlib (pip install matplotlib)

Seaborn (pip install seaborn)

Scikit-learn (pip install scikit-learn)

Pandas (pip install pandas)

Numpy (pip install numpy)


# Setup Instructions

# Clone or Download: 
Clone this repository or download the project files from the source.

# Install Dependencies: 
Install the required libraries by running  the command:

# You can install all dependencies using:
```

pip install tensorflow>=2.0
pip install keras>=2.0
pip install opencv-python>=4.5.0
pip install numpy>=1.19
pip install matplotlib>=3.2
pip install seaborn>=0.10.1
pip install scikit-learn>=0.24
pip install pandas>=1.1
pip install pickle5

```

# Dataset Preparation:

Ensure that your dataset is organized in folders where each folder is named after the label (e.g., Autistic and Non_Autistic).
Each folder should contain images representing the respective class.

# Running the Application:
Run the autism_detection.py script to launch the application.

# How to Use
# Step 1: Upload Autism Dataset

Click the "Upload Autism Dataset" button.

Select a folder that contains subfolders for each class (e.g., Autistic and Non_Autistic).

# Step 2: Preprocess the Dataset

Click "Preprocess Dataset". This will:

Load images from the dataset.

Resize them to 64x64 pixels.

Normalize pixel values to the range [0, 1].

Split the dataset into training (80%) and testing (20%) sets.

# Step 3: Train Models

Click "Run ResNet50 Algorithm" to train the ResNet50 model.

Click "Run Xception Algorithm" to train the Xception model.

Training will take time depending on the size of the dataset and your hardware.

# Step 4: View Model Performance

The application will show the accuracy, precision, recall, and F1-score for both models (ResNet50 and Xception).

The confusion matrix for each model will be displayed using a heatmap.

# Step 5: Make Predictions

Click "Predict Autism from Test Image" to select a test image.

The selected image will be classified, and the result will be displayed on the image as a text overlay (e.g., "Autistic Detected").

# Step 6: Generate Comparison Graph

Click "Comparison Graph" to generate a bar chart comparing the performance of the two models across the metrics (Accuracy, Precision, Recall, and F1-Score).

# Model Evaluation Metrics

# Accuracy: 
The percentage of correctly classified samples.
Precision: The proportion of true positive predictions out of all positive predictions.

# Recall: 
The proportion of true positives out of all actual positive samples.

# F1-Score: 
The harmonic mean of precision and recall. Useful for imbalanced datasets.

# Example Performance Metrics:
# ResNet50:
```
Accuracy: 85%

Precision: 88%

Recall: 80%

F1-Score: 84%
```

# Xception:
```
Accuracy: 87%
Precision: 90%
Recall: 83%
F1-Score: 86%
```
# Visualizations

# Confusion Matrix:
A heatmap will be generated to visually evaluate the performance of each model.

# Comparison Graph: 
A bar chart comparing the accuracy, precision, recall, and F1 scores of ResNet50 and Xception will be generated.

# Example Output

# Confusion Matrix (Heatmap):

Visual representation showing how well each model predicts each class (Autistic and Non_Autistic).
# Performance Metrics Table:
```
Algorithm	Accuracy    	Precision	  Recall	  F1-Score
ResNet50	85%	               88%	        80%	       84%
Xception	87%	               90%	        83%	       86%

```
# Future Enhancements

# Model Hyperparameter Tuning:
 Use techniques like Grid Search or Random Search to fine-tune model hyperparameters.

# Cross-Validation: 
Implement K-fold cross-validation to ensure more robust performance evaluation.

# Edge Deployment: 
Deploy the trained models on edge devices for real-time prediction.

# Advanced Data Augmentation: 
Introduce more advanced data augmentation techniques to improve model generalization.

# License
This project is licensed under the MIT License - see the LICENSE file for details.

