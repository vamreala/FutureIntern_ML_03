from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
from utils import preprocess_image

# Initialize Flask app
app = Flask(__name__)

# Load the trained model from the src folder
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "cifar10_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels for CIFAR-10 dataset
CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat",
    "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"
]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if an image was uploaded
        if "file" not in request.files:
            return "No file uploaded!", 400
        
        file = request.files["file"]
        if file.filename == "":
            return "No file selected!", 400
        
        # Process the uploaded image
        image = preprocess_image(file)
        prediction = model.predict(image)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction)

        return render_template(
            "index.html",
            prediction=predicted_class,
            confidence=confidence
        )
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
