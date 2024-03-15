# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 16:19:08 2022

@author: akash
"""

from flask import Flask,render_template,request,redirect
import sys
import pickle
import numpy as np

app = Flask(__name__,template_folder='templates')

#FEATURE
def feature(key,value):
    race_dict = {'Amer-Indian-Eskimo': 0.1157556270096463,
                 'Asian-Pac-Islander': 0.26564003849855633,
                 'Black': 0.12387964148527529,
                 'Other': 0.09225092250922509,
                 'White': 0.2558599367270636}
    
    sex_dict = {'Female': 0, 'Male': 1}
    
    marital_dict = {'Divorced': 0.10420886788206167,
                    'Married-AF-spouse': 0.43478260869565216,
                    'Married-civ-spouse': 0.4468482905982906,
                    'Married-spouse-absent': 0.08133971291866028,
                    'Never-married': 0.04596087241411589,
                    'Separated': 0.06439024390243903,
                    'Widowed': 0.08559919436052367}
    
    workclass_dict = {'State-gov': 0.27195685670261943,
                      'Self-emp-not-inc': 0.2849271940181031,
                      'Private': 0.21009293983368663,
                      'Federal-gov': 0.38645833333333335,
                      'Local-gov': 0.29479216435738176,
                      'Self-emp-inc': 0.557347670250896,
                      'Without-pay': 0.0}
    
    occupation_dict = {'Adm-clerical': 0.13448275862068965,
                       'Exec-managerial': 0.4840137727496311,
                       'Handlers-cleaners': 0.06277372262773723,
                       'Prof-specialty': 0.3426374728397125,
                       'Other-service': 0.04157814871016692,
                       'Sales': 0.2693150684931507,
                       'Craft-repair': 0.22664064405952672,
                       'Transport-moving': 0.20037570444583594,
                       'Farming-fishing': 0.11569416498993963,
                       'Machine-op-inspct': 0.12487512487512488,
                       'Tech-support': 0.30495689655172414,
                       'Protective-serv': 0.325115562403698,
                       'Armed-Forces': 0.1111111111111111,
                       'Priv-house-serv': 0.006711409395973154}
    
    country_dict = {'United-States': 0.24592478069438375,
                    'Cuba': 0.2631578947368421,
                    'Jamaica': 0.12345679012345678,
                    'India': 0.4,
                    'Mexico': 0.05132192846034215,
                    'South': 0.2,
                    'Puerto-Rico': 0.10526315789473684,
                    'Honduras': 0.07692307692307693,
                    'England': 0.3333333333333333,
                    'Canada': 0.32231404958677684,
                    'Germany': 0.32116788321167883,
                    'Iran': 0.4186046511627907,
                    'Philippines': 0.30808080808080807,
                    'Italy': 0.3424657534246575,
                    'Poland': 0.03389830508474576,
                    'Columbia': 0.3684210526315789,
                    'Cambodia': 0.16666666666666666,
                    'Thailand': 0.14285714285714285,
                    'Ecuador': 0.1111111111111111,
                    'Laos': 0.39215686274509803,
                    'Taiwan': 0.09090909090909091,
                    'Haiti': 0.10810810810810811,
                    'Portugal': 0.02857142857142857,
                    'Dominican-Republic': 0.08490566037735849,
                    'El-Salvador': 0.41379310344827586,
                    'France': 0.046875,
                    'Guatemala': 0.26666666666666666,
                    'China': 0.3870967741935484,
                    'Japan': 0.375,
                    'Yugoslavia': 0.06451612903225806,
                    'Peru': 0.0,
                    'Outlying-US(Guam-USVI-etc)': 0.25,
                    'Scotland': 0.27586206896551724,
                    'Trinadad&Tobago': 0.058823529411764705,
                    'Greece': 0.07462686567164178,
                    'Nicaragua': 0.3, 
                    'Vietnam': 0.20833333333333334,
                    'Hong': 0.23076923076923078}
    
    if key == "Race":
        if value in race_dict:
            return race_dict[value]
    elif key == "Gender":
        if value in sex_dict:
            return sex_dict[value]
    elif key == "Marital - Status":
        if value in marital_dict:
            return marital_dict[value]
    elif key == "Workclass":
        if value in workclass_dict:
            return workclass_dict[value]
    elif key == "Occupation":
        if value in occupation_dict:
            return occupation_dict[value]
    else:
        if value in country_dict:
            return country_dict[value]

#IMPORT MODEL
model = pickle.load(open('model.pkl','rb'))

@app.route("/",methods = ['GET'])
def model_ux():
    return render_template('Flasktry.html')

@app.route("/predict",methods = ['POST'])
def predict():
    if request.method == "POST":
        result = request.form
        arr = []
        for key,value in result.items():
            if key in ['Age','Education - Num','Capital Gain','Capital Loss','Working - Hours']:
                arr.append(int(value))
            else:
                val = feature(key,value)
                arr.append(val)
        final = [np.array(arr).astype(dtype = "float32")]
        predicted = model.predict(final)
        if predicted[0] == 0:
            out = "The Person's Salary is Less than 50K"
        else:
           out = "The Person's Salary is More than 50K"
        return render_template('Flasktry.html',result_text = out)
    
    
if __name__ == '__main__':
    app.run(host = '0.0.0.0',port = 8080)