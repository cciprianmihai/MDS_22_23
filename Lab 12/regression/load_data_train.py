import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
import pickle

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Use sepal length as input feature and petal length as target variable
X = df[['sepal length (cm)']]
y = df['petal length (cm)']

# Save the dataset to a CSV file
df.to_csv('iris_dataset.csv', index=False)

# Load the dataset
df = pd.read_csv('iris_dataset.csv')
X = df[['sepal length (cm)']]
y = df['petal length (cm)']

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Save the trained model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

