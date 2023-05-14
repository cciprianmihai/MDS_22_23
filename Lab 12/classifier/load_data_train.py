import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
import pickle
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten the input images
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Normalize the pixel values to the range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# Save the preprocessed data to separate files
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

# Load the preprocessed data from the NPY files
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

# Train the classifier
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# Save the trained classifier to a file
with open('classifier.pkl', 'wb') as file:
    pickle.dump(classifier, file)