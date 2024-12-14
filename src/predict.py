import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.data_preprocessing import load_data

def make_prediction(image_idx=0):
    # Load the trained model
    model = tf.keras.models.load_model('cifar10_model.h5')

    # Load data
    _, _, test_images, test_labels = load_data()

    # Pick an image from the test set
    image = test_images[image_idx]
    true_label = test_labels[image_idx]

    # Make a prediction
    prediction = model.predict(np.expand_dims(image, axis=0))
    predicted_label = np.argmax(prediction)

    # Plot the image and the prediction
    plt.imshow(image)
    plt.title(f"True: {true_label}, Pred: {predicted_label}")
    plt.show()

if __name__ == "__main__":
    make_prediction()
