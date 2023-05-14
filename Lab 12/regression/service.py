import pickle
from flask import Flask, request, jsonify

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json
    sepal_length = data['sepal_length']

    # Perform prediction using the trained model
    prediction = model.predict([[sepal_length]])

    # Create a JSON response with the prediction
    response = {'prediction': prediction[0]}
    return jsonify(response)

if __name__ == '__main__':
    app.run()
