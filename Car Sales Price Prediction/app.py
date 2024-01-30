from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained neural network model
model = load_model('ann_model.h5')  # Update with your actual model file

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        features = [float(request.form['annual Salary']),
                    float(request.form['credit card debt']),
                    float(request.form['net worth']),
                    float(request.form['age'])]

        # Standardize the input features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform([features])

        # Make a prediction
        prediction = model.predict(features_scaled)

        # Display the prediction on the result page
        return render_template('result.html', prediction=prediction[0][0])

if __name__ == '__main__':
    app.run(debug=True)