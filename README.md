# CIFAR-10 Image Classification

This project is a web application for classifying images into one of the 10 classes in the CIFAR-10 dataset using a pre-trained deep learning model.

# Folder Structure

cifar-10-classification/
├── src/
│   ├── train.py               # Script for training the model
│   ├── evaluate.py            # Script for evaluating the model
│   ├── preprocess.py          # Image preprocessing script
│   ├── cifar10_model.h5       # Pre-trained model file
│   ├── deployment/            # Flask app folder
│   │   ├── app.py             # Flask backend
│   │   ├── templates/
│   │   │   └── index.html     # Frontend HTML
│   │   ├── static/
│   │   │   └── styles.css     # CSS styles
│   │   ├── utils.py           # Helper functions for the app
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation

# Features

Pre-trained model for CIFAR-10 image classification.

Web interface to upload and classify images.

Returns the predicted class and confidence score for each image.

# Installation

Clone the RepositoryClone this repository to your local machine:

git clone <repository_url>
cd cifar-10-classification

Install DependenciesInstall the required Python dependencies listed in requirements.txt:

pip install -r requirements.txt

Run the Flask AppNavigate to the src/deployment directory and start the Flask application:

cd src/deployment
python app.py

Open the ApplicationOpen your browser and go to http://127.0.0.1:5000/ to access the web app.

# Usage

Upload an image (32x32 pixels, as required by CIFAR-10).

View the predicted class and confidence score on the results page.

# Dependencies

Flask

TensorFlow

NumPy

Pillow

# About the CIFAR-10 Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are:

1. Airplane

2. Automobile

3. Bird

4. Cat

5. Deer

6. Dog

7. Frog

8. Horse

9. Ship

10. Truck

11. License

This project is licensed under the MIT License. See the LICENSE file for details.

