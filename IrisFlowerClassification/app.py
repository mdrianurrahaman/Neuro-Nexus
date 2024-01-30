# app.py
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained machine learning model
model = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Make a prediction using the loaded model
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)[0]

        # Map the numeric prediction to the corresponding Iris species
        species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        predicted_species = species_mapping[prediction]

        return render_template('result.html', species=predicted_species)

if __name__ == '__main__':
    app.run(debug=True)
