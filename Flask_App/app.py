#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:22:27 2019

@author: danielobennett
"""

'''
Import libraries
'''

from flask import Flask, render_template, request

import pandas as pd
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
#for securing files before upload



'''
ACTUAL FLASK CODE
'''

app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = '/Users/danielobennett/metis/work/projects/Project05/test_songs/'
#app.secret_key = 'some secret key'

#model = load in model with pickle

with open("lg.pkl", "rb") as f:
    lg_model = pickle.load(f)

with open("scaler.pkl","rb") as g:
    scaler = pickle.load(g)

#homepage
@app.route('/')
def home():
	return render_template('home.html')




#results of lytrical recommendation page
@app.route('/results', methods=['GET', 'POST']) #is post since we're retreiving...
def predict():
    if request.method == 'POST':
        funding = int(request.form['amount_funding'])
        rounds = int(request.form['rounds_funding'])
        months = int(request.form['months_funding'])
        industry = request.form['industry']
        country = request.form['country']
        state = request.form['state']

        #create a 1 row dataframe
        new_data = {}

        new_data['Funding_total_usd'] = funding
        new_data['Funding_rounds']    = rounds
        new_data['Months_until_1st_fundings'] = months

        # industry
        if industry == 'Biotechnology':
            new_data['Biotechnology'] = 1
        else:
            new_data['Biotechnology'] = 0
    
        if industry == 'Clean Technology':
            new_data['Clean Technology'] = 1
        else:
            new_data['Clean Technology'] = 0
       
        if industry == 'Curated Web':
            new_data['Curated Web'] = 1
        else:
            new_data['Curated Web'] = 0
    
        if industry == 'Enterprise Software':
            new_data['Enterprise Software'] = 1
        else:
            new_data['Enterprise Software'] = 0
    
        if industry == 'Games':
            new_data['Games'] = 1
        else:
            new_data['Games'] = 0
        
        if industry == 'Hardware + Software':
            new_data['Hardware + Software'] = 1
        else:
            new_data['Hardware + Software'] = 0

        if industry == 'Mobile':
            new_data['Mobile'] = 1
        else:
            new_data['Mobile'] = 0
    
        if industry == 'Security':
            new_data['Security'] = 1
        else:
            new_data['Security'] = 0
    
        if industry == 'Social Media':
            new_data['Social Media'] = 1
        else:
            new_data['Social Media'] = 0
    
        if industry == 'Security':
            new_data['Security'] = 1
        else:
            new_data['Security'] = 0
    
        if industry == 'Software':
            new_data['Software'] = 1
        else:
            new_data['Software'] = 0

    
        # Country     
        if country == 'GBR':
            new_data['GBR'] = 1
        else:
            new_data['GBR'] = 0

        if country == 'ISR':
            new_data['ISR'] = 1
        else:
            new_data['ISR'] = 0
            
        if country == 'RUS':
            new_data['RUS'] = 1
        else:
            new_data['RUS'] = 0

        if country == 'USA':
            new_data['USA'] = 1
        else:
            new_data['USA'] = 0
            
        # State  
        if state == 'CA':
            new_data['CA'] = 1
        else:
            new_data['CA'] = 0
            
        if state == 'MA':
            new_data['MA'] = 1
        else:
            new_data['MA'] = 0
            
        if state == 'NY':
            new_data['NY'] = 1
        else:
            new_data['NY'] = 0

        #convert to data farme
        new_data = pd.DataFrame.from_dict(new_data,  orient='index').transpose()

        '''Add a function that combines those inputs, together with all of the other columns that are needed
        for the input to the model (ie should have same amount if columns as X_train)'''
        #Function goes here
        #loaded in model
        #prob = model.predict(new_data)
        #if prob > threshold:
         #   result = 'Profitable'
        #else:
         #   result = "Not Profitable"
        #return render_template('results.html', results = result )
        x_scaled = scaler.transform(np.array(new_data).reshape(1, -1))

        ###OR
        prob = lg_model.predict_proba(x_scaled)[:,1]
        return render_template('results.html', results = format(prob[0],'.2%') )  

if __name__ == '__main__':
	app.run(debug=False)
#	app.run(debug=True)
