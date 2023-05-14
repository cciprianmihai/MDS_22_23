import pickle
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io

# Load the trained classifier
with open('classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    image_data = request.files['image'].read()
    image = Image.open(io.BytesIO(image_data))

    # Preprocess the image
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to MNIST size
    image = np.array(image)  # Convert to numpy array
    image = image / 255.0  # Normalize pixel values
    image = image.reshape(1, -1)  # Flatten the image

    # Perform prediction using the trained classifier
    prediction = classifier.predict(image)

    # Convert numpy.uint8 to int
    prediction = int(prediction[0])

    # Create a JSON response with the prediction
    response = {'prediction': prediction}
    return jsonify(response)

if __name__ == '__main__':
    app.run()
