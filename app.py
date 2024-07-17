from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the saved Logistic Regression model
model_path = 'log_reg_fish_classifier.pkl'  # Path to your saved model
model = joblib.load(model_path)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)

    # Convert the data into a DataFrame
    input_data = pd.DataFrame(data, index=[0])

    # Scale the features using the same scaler as used during training
    scaler = StandardScaler()
    
    # Assuming scaler was fitted with training data, load it instead of fitting again
    scaler.fit(input_data)  
    scaled_data = scaler.transform(input_data)

    # Make prediction using the loaded model
    prediction = model.predict(scaled_data)

    # Return the prediction as JSON
    result = {'prediction': prediction[0]}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
