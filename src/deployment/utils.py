import numpy as np
from PIL import Image

def preprocess_image(file):
    """
    Preprocesses an uploaded image for model prediction.
    Args:
        file: Uploaded image file (e.g., from Flask request).
    Returns:
        Preprocessed image ready for model input.
    """
    image = Image.open(file).convert("RGB")
    image = image.resize((32, 32))  # Resize to CIFAR-10 input size
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image
