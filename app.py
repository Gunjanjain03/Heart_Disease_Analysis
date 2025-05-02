from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Initialize the Flask app
app = Flask(__name__)

# Load the dataset (assuming it's in the same directory)
heart_data = pd.read_csv('heart.csv')

# Data preprocessing (this can be done once at the start)
heart_data.drop_duplicates(inplace=True)
x = heart_data.drop(columns='target', axis=1)
y = heart_data['target']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Train the model (only once when the app starts)
model = LogisticRegression()
model.fit(x_train, y_train)

# Calculate accuracy on the testing data
y_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, y_test_prediction)

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    input_data = request.json['input']
    
    # Convert input data to numpy array and reshape it for prediction
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Make a prediction
    prediction = model.predict(input_data_reshaped)
    
    # Return result
    if prediction[0] == 0:
        result = 'The person does not have heart disease'
    else:
        result = 'The person has heart disease'
    
    # Return the result as a JSON response
    return jsonify({'prediction': result})

# Define a route to check model accuracy on test data
@app.route('/accuracy', methods=['GET'])
def get_accuracy():
    return jsonify({'test_data_accuracy': test_data_accuracy})

# Define a route to check API status
@app.route('/', methods=['GET'])
def home():
    return "Heart Disease Prediction API is running."

# Run the app
if __name__ == '__main__':
    app.run(debug=True)