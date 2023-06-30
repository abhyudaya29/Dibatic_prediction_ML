import pickle
from flask import Flask,render_template,jsonify,request

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app=Flask(__name__)
model=pickle.load(open('models/clf.pkl','rb'))
scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Glucose=float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness= float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI= float(request.form.get('BMI'))
        Age= float(request.form.get('Age'))
        
        new_data=scaler.transform([[Glucose,BloodPressure,SkinThickness,Insulin, BMI,Age]])
        result=model.predict(new_data)

        return render_template('home.html',result=result[0])
    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")
