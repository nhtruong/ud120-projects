#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

POI = {person:enron_data[person] for person in enron_data if enron_data[person]['poi'] == True}

enron_data['PRENTICE JAMES']['total_stock_value']

enron_data['COLWELL WESLEY']['from_this_person_to_poi']

enron_data['SKILLING JEFFREY K']['total_payments']
enron_data['LAY KENNETH L']['total_payments']
enron_data['FASTOW ANDREW S']['total_payments']

from sklearn import linear_model
reg = linear_model.LinearRegression()

