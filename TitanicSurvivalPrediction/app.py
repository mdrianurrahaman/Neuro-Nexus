from flask import Flask,render_template,request

import pandas as pd
import pickle 
import numpy as np 
import os
app = Flask(__name__)

model=pickle.load(open("trained_model.pkl","rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def prediction_datapoint():
    Pclass=int(request.form["Pclass"])
    Sex=int(request.form["Sex"])
    Age=float(request.form["Age"])
    SibSp = int(request.form["SibSp"])
    Parch = int(request.form["Parch"])
    Fare=float(request.form["Fare"])
    Embarked=int(request.form["Embarked"])

    input_data=(Pclass,Sex,Age,SibSp,Parch,Fare,Embarked)
    input_data_as_np_array=np.asarray(input_data)
    input_data_reshaped=input_data_as_np_array.reshape(1,-1)
    prediction=model.predict(input_data_reshaped)
    prediction[0]
    if prediction[0] == 0 :
       prediction="not survived"
    else:
       prediction="survived"  
    return render_template("result.html",final_result=prediction) 

if __name__ == '__main__':
    app.run(debug=True)
