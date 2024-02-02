from flask import Flask,render_template,request
import pickle

app = Flask(__name__)

# Load the pre-trained machine learning model
model = pickle.load(open('trained_model.pkl', 'rb')) # Replace with your actual model filename

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('form.html')


    else: 
        # Extract features from the form or request data
        year = int(request.form['year'])
        duration = int(request.form['duration'])
        votes = int(request.form['votes'])
        directors = int(request.form['directors'])
        genres = int(request.form['genres'])
        actors = int(request.form['actors'])

        # Ensure the features match the expected input shape
        # You may need to preprocess the features as per your model requirements

        # Make the prediction
        prediction = model.predict([[year, duration, votes, directors, genres, actors]])

        # Return the prediction to the user
        return render_template('result.html', prediction=prediction[0])
    
if __name__ == '__main__':
    app.run(debug=True)
